#include <set>
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLoc.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace cl;
using namespace llvm::object;

static list<std::string>
    InputFilenames(Positional, desc("<input object files or .dSYM bundles>"),
                   OneOrMore);

static opt<bool>
    CompareThings("compare", desc("Compare these two files pls"),
                   cl::Optional);
static opt<std::string>
    ContainsVar("contains", desc("Is this variable name in this file?"),
                   cl::Optional);


std::map<std::string, size_t> AllTheVariablesSeen;

void collectRec(DWARFDie &Block, Twine Path);

static void error(StringRef Prefix, std::error_code EC) {
  if (!EC)
    return;
  WithColor::error() << Prefix << ": " << EC.message() << "\n";
  exit(1);
}

llvm::Optional<std::string> getSubprogramName(DWARFDie &SP) {
  // Name is immediately available.
  auto LinkName = SP.find(dwarf::DW_AT_linkage_name);
  if (LinkName)
    return std::string(*LinkName->getAsCString());

  // There's a definition somewhere else we're referring to.
  auto Spec = SP.find(dwarf::DW_AT_specification);
  if (Spec) {
    auto SpecOffs = Spec->getAsRelativeReference();
    DWARFDie SpecDIE = SpecOffs->Unit->getDIEForOffset(SpecOffs->Offset);
    return getSubprogramName(SpecDIE);
  }

  // We might just hav ea plain, c-like name.
  auto Name = SP.find(dwarf::DW_AT_name);
  if (Name)
    return std::string(*Name->getAsCString());

  // We might encounter a subprogram that's the instance of an inlined decl.
  auto Origin = SP.find(dwarf::DW_AT_abstract_origin);
  if (Origin) {
    auto OriginOffs = Origin->getAsRelativeReference();
    DWARFDie OriginDIE = OriginOffs->Unit->getDIEForOffset(OriginOffs->Offset);
    return getSubprogramName(OriginDIE);
  }

  // LLVM appears to do this. It doesn't seem wise, but it happens.
  //errs() << "Failed to find subprogram name, lol\n";
  return None;
}

void collectSubprogram(DWARFDie &SP, Twine Path) {
  // Ignore the "inlined" abstract description of a subprogram. 
  auto Inlined = SP.find(dwarf::DW_AT_inline);
  if (Inlined)
    return;

  auto Name = getSubprogramName(SP);
  if (!Name)
    return;

  collectRec(SP, Path + Twine("/sp:") + Twine(*Name));
}

void collectInlinedSP(DWARFDie &Inlined, Twine Path) {
  auto Origin = Inlined.find(dwarf::DW_AT_abstract_origin);
  assert(Origin);
  if (Origin->getForm() != dwarf::DW_FORM_ref4) {
    errs() << "Expected ref4 for all abstract origins\n";
    exit(1);
  }

  auto Offs = Origin->getAsRelativeReference();
  assert(Offs);
  DWARFDie OriginDIE = Offs->Unit->getDIEForOffset(Offs->Offset);
  // XXX, what happens if it's an illegal offset?

  auto Name = getSubprogramName(OriginDIE);
  if (!Name)
    return;

  unsigned long CallLine = 0, CallColumn = 0;
  auto LineAttr = Inlined.find(dwarf::DW_AT_call_line);
  assert(LineAttr);
  CallLine = *LineAttr->getAsUnsignedConstant();

  // This is optional -- call site might have had its DebugLoc dropped.
  auto ColumnAttr = Inlined.find(dwarf::DW_AT_call_column);
  if (ColumnAttr)
    CallColumn = *ColumnAttr->getAsUnsignedConstant();

  // XXX do something with those thigns

  collectRec(Inlined, Path + Twine("/sp-inlined:") + Twine(*Name));
}

void collectVariable(DWARFDie &Var, Twine Path) {
  auto Location = Var.find(dwarf::DW_AT_location);
  if (!Location)
    return;

  // Calculate some location faff,
  Expected<std::vector<DWARFLocationExpression>> Loc = Var.getLocations(dwarf::DW_AT_location);
  if (!Loc)
    return;

  size_t size = 0;
  if (Loc->size() == 1 && !(*Loc)[0].Range) {
    // It's a single location expression. Just assign a large number -- we'll
    // compare equal with other single location expression, and anything that
    // produces a location list will produce less.
    size = 99999999;
  } else {
    for (auto &Entry : *Loc) {
      assert(Entry.Range);
      uint64_t sz = Entry.Range->HighPC - Entry.Range->LowPC;
      size += sz;
    }
  }

  std::string Name;
  if (auto NameAttr = Var.find(dwarf::DW_AT_name)) {
    Name = std::string(*NameAttr->getAsCString());
  } else if (auto OriginAttr = Var.find(dwarf::DW_AT_abstract_origin)) {
    auto OriginOffs = OriginAttr->getAsRelativeReference();
    DWARFDie OriginDIE = OriginOffs->Unit->getDIEForOffset(OriginOffs->Offset);
    auto NameAttr = OriginDIE.find(dwarf::DW_AT_name);
    if (!NameAttr)
      // Apparently permitted, see LICMPass::run for an un-named param
      return;
    Name = std::string(*NameAttr->getAsCString());
  } else {
    // Un-named, non-abstract variable. Again, C++ appears to permit this.
    return;
  }

  auto FullName = Path + Twine("/var:") + Name;
  AllTheVariablesSeen.insert(std::make_pair(FullName.str(), size));
}

void collectRec(DWARFDie &Block, Twine Path) {
  unsigned int NumBlocks = 0;
  DWARFDie Child = Block.getFirstChild();
  while (Child) {
    const dwarf::Tag Tag = Child.getTag();
    if (Tag == dwarf::DW_TAG_lexical_block) {
      // XXX do something with block counts
      collectRec(Child, Path + "/block-" + Twine(NumBlocks) + Twine(":"));
      ++NumBlocks;
    } else if (Tag == dwarf::DW_TAG_inlined_subroutine) {
      collectInlinedSP(Child, Path);
    } else if (Tag == dwarf::DW_TAG_variable) {
      collectVariable(Child, Path);
    } else if (Tag == dwarf::DW_TAG_formal_parameter) {
      collectVariable(Child, Path);
    }
    Child = Child.getSibling();
  }
}

void collectedNestedSubprogramsRec(DWARFDie Block, Twine Path) {
  // Seek out top level subprograms.
  DWARFDie Child = Block.getFirstChild();
  while (Child) {
    const dwarf::Tag Tag = Child.getTag();
    if (Tag == dwarf::DW_TAG_subprogram)
      collectSubprogram(Child, Path);
    else
      collectedNestedSubprogramsRec(Child, Path); // TODO: Add to Path string.
    Child = Child.getSibling();
  }
}

void frobThisFile(std::string FileName) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BuffOrErr =
    MemoryBuffer::getFileOrSTDIN(FileName);
  error(FileName, BuffOrErr.getError());
  std::unique_ptr<MemoryBuffer> Buffer = std::move(BuffOrErr.get());

  Expected<std::unique_ptr<Binary>> BinOrErr = object::createBinary(*Buffer.get());
  error("Creating binary", errorToErrorCode(BinOrErr.takeError()));

  auto *Obj = dyn_cast<ObjectFile>(BinOrErr->get());
  if (!Obj) {
    errs() << "Input buffer not an object file?\n";
    exit(1);
  }

  std::unique_ptr<DWARFContext> DICtx = DWARFContext::create(*Obj);
  if (!DICtx.get()) {
    errs() << "No DWARF info in object file?\n";
    exit(1);
  }

  for (const auto &CU : DICtx->compile_units()) {
    DWARFDie CUDie = CU->getNonSkeletonUnitDIE(false);
    if (!CUDie)
      continue;

    auto Name = CUDie.find(dwarf::DW_AT_name);
    assert(Name);
    auto CUName = std::string(*Name->getAsCString());

    // Seek out top level subprograms.
    DWARFDie Child = CUDie.getFirstChild();
    while (Child) {
      if (Child.getTag() == dwarf::DW_TAG_subprogram)
        collectSubprogram(Child, Twine(CUName));
      else      
        collectedNestedSubprogramsRec(Child, Twine(CUName) + Twine(":"));
      Child = Child.getSibling();
    }
  }
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  errs().tie(&outs());

  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();

  cl::ParseCommandLineOptions(
      argc, argv,
      "Catch fire and cover the world in bees\n");

  if (ContainsVar != "") {
    frobThisFile(InputFilenames[0]);
    auto It = AllTheVariablesSeen.find(ContainsVar);
    if (It == AllTheVariablesSeen.end())
      exit(1);
    exit(0);
  }

  if (!CompareThings) {
    frobThisFile(InputFilenames[0]);

    for (auto Name : AllTheVariablesSeen) {
      outs() << Name.first << " " << Name.second <<"\n";
    }

    exit(0);
  }

  if (InputFilenames.size() != 2) {
    errs() << "Two files needed in compare mode\n";
    exit(1);
  }

  outs() << "Variables that are in the first file, but not the second:\n";

  frobThisFile(InputFilenames[0]);
  std::set<std::string> OldNames;
  for (auto &Entry : AllTheVariablesSeen)
    OldNames.insert(Entry.first);

  AllTheVariablesSeen = decltype(AllTheVariablesSeen)();
  frobThisFile(InputFilenames[1]);

  std::set<std::string> NewNames;
  for (auto &Entry : AllTheVariablesSeen)
    NewNames.insert(Entry.first);

  std::vector<std::string> Diff;
  auto Inserter = std::back_insert_iterator<decltype(Diff)>(Diff);
  std::set_difference(OldNames.begin(), OldNames.end(),
                      NewNames.begin(), NewNames.end(),
                      Inserter);

  for (auto &Elem : Diff)
    outs() << Elem << "\n";
  
  return 0;
}
