set(files
  __algorithm/adjacent_find.h
  __algorithm/all_of.h
  __algorithm/any_of.h
  __algorithm/binary_search.h
  __algorithm/clamp.h
  __algorithm/comp_ref_type.h
  __algorithm/comp.h
  __algorithm/copy_backward.h
  __algorithm/copy_if.h
  __algorithm/copy_n.h
  __algorithm/copy.h
  __algorithm/count_if.h
  __algorithm/count.h
  __algorithm/equal_range.h
  __algorithm/equal.h
  __algorithm/fill_n.h
  __algorithm/fill.h
  __algorithm/find_end.h
  __algorithm/find_first_of.h
  __algorithm/find_if_not.h
  __algorithm/find_if.h
  __algorithm/find.h
  __algorithm/for_each_n.h
  __algorithm/for_each.h
  __algorithm/generate_n.h
  __algorithm/generate.h
  __algorithm/half_positive.h
  __algorithm/includes.h
  __algorithm/inplace_merge.h
  __algorithm/is_heap_until.h
  __algorithm/is_heap.h
  __algorithm/is_partitioned.h
  __algorithm/is_permutation.h
  __algorithm/is_sorted_until.h
  __algorithm/is_sorted.h
  __algorithm/iter_swap.h
  __algorithm/lexicographical_compare.h
  __algorithm/lower_bound.h
  __algorithm/make_heap.h
  __algorithm/max_element.h
  __algorithm/max.h
  __algorithm/merge.h
  __algorithm/min_element.h
  __algorithm/min.h
  __algorithm/minmax_element.h
  __algorithm/minmax.h
  __algorithm/mismatch.h
  __algorithm/move_backward.h
  __algorithm/move.h
  __algorithm/next_permutation.h
  __algorithm/none_of.h
  __algorithm/nth_element.h
  __algorithm/partial_sort_copy.h
  __algorithm/partial_sort.h
  __algorithm/partition_copy.h
  __algorithm/partition_point.h
  __algorithm/partition.h
  __algorithm/pop_heap.h
  __algorithm/prev_permutation.h
  __algorithm/push_heap.h
  __algorithm/remove_copy_if.h
  __algorithm/remove_copy.h
  __algorithm/remove_if.h
  __algorithm/remove.h
  __algorithm/replace_copy_if.h
  __algorithm/replace_copy.h
  __algorithm/replace_if.h
  __algorithm/replace.h
  __algorithm/reverse_copy.h
  __algorithm/reverse.h
  __algorithm/rotate_copy.h
  __algorithm/rotate.h
  __algorithm/sample.h
  __algorithm/search_n.h
  __algorithm/search.h
  __algorithm/set_difference.h
  __algorithm/set_intersection.h
  __algorithm/set_symmetric_difference.h
  __algorithm/set_union.h
  __algorithm/shift_left.h
  __algorithm/shift_right.h
  __algorithm/shuffle.h
  __algorithm/sift_down.h
  __algorithm/sort_heap.h
  __algorithm/sort.h
  __algorithm/stable_partition.h
  __algorithm/stable_sort.h
  __algorithm/swap_ranges.h
  __algorithm/transform.h
  __algorithm/unique_copy.h
  __algorithm/unique.h
  __algorithm/unwrap_iter.h
  __algorithm/upper_bound.h
  __availability
  __bit_reference
  __bits
  __bsd_locale_defaults.h
  __bsd_locale_fallbacks.h
  __config
  __debug
  __errc
  __format/format_error.h
  __format/format_parse_context.h
  __function_like.h
  __functional_base
  __functional/binary_function.h
  __functional/binary_negate.h
  __functional/bind_front.h
  __functional/bind.h
  __functional/binder1st.h
  __functional/binder2nd.h
  __functional/default_searcher.h
  __functional/function.h
  __functional/hash.h
  __functional/identity.h
  __functional/invoke.h
  __functional/is_transparent.h
  __functional/mem_fn.h
  __functional/mem_fun_ref.h
  __functional/not_fn.h
  __functional/operations.h
  __functional/perfect_forward.h
  __functional/pointer_to_binary_function.h
  __functional/pointer_to_unary_function.h
  __functional/ranges_operations.h
  __functional/reference_wrapper.h
  __functional/unary_function.h
  __functional/unary_negate.h
  __functional/unwrap_ref.h
  __functional/weak_result_type.h
  __hash_table
  __iterator/access.h
  __iterator/advance.h
  __iterator/back_insert_iterator.h
  __iterator/concepts.h
  __iterator/data.h
  __iterator/default_sentinel.h
  __iterator/distance.h
  __iterator/empty.h
  __iterator/erase_if_container.h
  __iterator/front_insert_iterator.h
  __iterator/incrementable_traits.h
  __iterator/insert_iterator.h
  __iterator/istream_iterator.h
  __iterator/istreambuf_iterator.h
  __iterator/iter_move.h
  __iterator/iter_swap.h
  __iterator/iterator_traits.h
  __iterator/iterator.h
  __iterator/move_iterator.h
  __iterator/next.h
  __iterator/ostream_iterator.h
  __iterator/ostreambuf_iterator.h
  __iterator/prev.h
  __iterator/projected.h
  __iterator/readable_traits.h
  __iterator/reverse_access.h
  __iterator/reverse_iterator.h
  __iterator/size.h
  __iterator/wrap_iter.h
  __libcpp_version
  __locale
  __memory/addressof.h
  __memory/allocation_guard.h
  __memory/allocator_arg_t.h
  __memory/allocator_traits.h
  __memory/allocator.h
  __memory/auto_ptr.h
  __memory/compressed_pair.h
  __memory/construct_at.h
  __memory/pointer_safety.h
  __memory/pointer_traits.h
  __memory/raw_storage_iterator.h
  __memory/shared_ptr.h
  __memory/temporary_buffer.h
  __memory/uninitialized_algorithms.h
  __memory/unique_ptr.h
  __memory/uses_allocator.h
  __mutex_base
  __node_handle
  __nullptr
  __random/uniform_int_distribution.h
  __ranges/access.h
  __ranges/all.h
  __ranges/concepts.h
  __ranges/copyable_box.h
  __ranges/data.h
  __ranges/drop_view.h
  __ranges/empty_view.h
  __ranges/empty.h
  __ranges/enable_borrowed_range.h
  __ranges/enable_view.h
  __ranges/ref_view.h
  __ranges/size.h
  __ranges/subrange.h
  __ranges/transform_view.h
  __ranges/view_interface.h
  __split_buffer
  __std_stream
  __string
  __support/android/locale_bionic.h
  __support/fuchsia/xlocale.h
  __support/ibm/gettod_zos.h
  __support/ibm/limits.h
  __support/ibm/locale_mgmt_aix.h
  __support/ibm/locale_mgmt_zos.h
  __support/ibm/nanosleep.h
  __support/ibm/support.h
  __support/ibm/xlocale.h
  __support/musl/xlocale.h
  __support/newlib/xlocale.h
  __support/nuttx/xlocale.h
  __support/openbsd/xlocale.h
  __support/solaris/floatingpoint.h
  __support/solaris/wchar.h
  __support/solaris/xlocale.h
  __support/win32/limits_msvc_win32.h
  __support/win32/locale_win32.h
  __support/xlocale/__nop_locale_mgmt.h
  __support/xlocale/__posix_l_fallback.h
  __support/xlocale/__strtonum_fallback.h
  __threading_support
  __tree
  __tuple
  __undef_macros
  __utility/__decay_copy.h
  __utility/as_const.h
  __utility/cmp.h
  __utility/declval.h
  __utility/exchange.h
  __utility/forward.h
  __utility/in_place.h
  __utility/integer_sequence.h
  __utility/move.h
  __utility/pair.h
  __utility/piecewise_construct.h
  __utility/rel_ops.h
  __utility/swap.h
  __utility/to_underlying.h
  __variant/monostate.h
  algorithm
  any
  array
  atomic
  barrier
  bit
  bitset
  cassert
  ccomplex
  cctype
  cerrno
  cfenv
  cfloat
  charconv
  chrono
  cinttypes
  ciso646
  climits
  clocale
  cmath
  codecvt
  compare
  complex
  complex.h
  concepts
  condition_variable
  csetjmp
  csignal
  cstdarg
  cstdbool
  cstddef
  cstdint
  cstdio
  cstdlib
  cstring
  ctgmath
  ctime
  ctype.h
  cwchar
  cwctype
  deque
  errno.h
  exception
  execution
  experimental/__config
  experimental/__memory
  experimental/algorithm
  experimental/coroutine
  experimental/deque
  experimental/filesystem
  experimental/forward_list
  experimental/functional
  experimental/iterator
  experimental/list
  experimental/map
  experimental/memory_resource
  experimental/propagate_const
  experimental/regex
  experimental/set
  experimental/simd
  experimental/string
  experimental/type_traits
  experimental/unordered_map
  experimental/unordered_set
  experimental/utility
  experimental/vector
  ext/__hash
  ext/hash_map
  ext/hash_set
  fenv.h
  filesystem
  float.h
  format
  forward_list
  fstream
  functional
  future
  initializer_list
  inttypes.h
  iomanip
  ios
  iosfwd
  iostream
  istream
  iterator
  latch
  limits
  limits.h
  list
  locale
  locale.h
  map
  math.h
  memory
  module.modulemap
  mutex
  new
  numbers
  numeric
  optional
  ostream
  queue
  random
  ranges
  ranges
  ratio
  regex
  scoped_allocator
  semaphore
  set
  setjmp.h
  shared_mutex
  span
  sstream
  stack
  stdbool.h
  stddef.h
  stdexcept
  stdint.h
  stdio.h
  stdlib.h
  streambuf
  string
  string_view
  string.h
  strstream
  system_error
  tgmath.h
  thread
  tuple
  type_traits
  typeindex
  typeinfo
  unordered_map
  unordered_set
  utility
  valarray
  variant
  vector
  version
  wchar.h
  wctype.h
  )

configure_file("__config_site.in" "${LIBCXX_GENERATED_INCLUDE_TARGET_DIR}/__config_site" @ONLY)

set(_all_includes "${LIBCXX_GENERATED_INCLUDE_TARGET_DIR}/__config_site")
foreach(f ${files})
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/${f}")
  set(dst "${LIBCXX_GENERATED_INCLUDE_DIR}/${f}")
  add_custom_command(OUTPUT ${dst}
    DEPENDS ${src}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
    COMMENT "Copying CXX header ${f}")
  list(APPEND _all_includes "${dst}")
endforeach()

add_custom_target(generate-cxx-headers ALL DEPENDS ${_all_includes})

add_library(cxx-headers INTERFACE)
add_dependencies(cxx-headers generate-cxx-headers ${LIBCXX_CXX_ABI_HEADER_TARGET})
# TODO: Use target_include_directories once we figure out why that breaks the runtimes build
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC" OR "${CMAKE_CXX_SIMULATE_ID}" STREQUAL "MSVC")
  target_compile_options(cxx-headers INTERFACE /I${LIBCXX_GENERATED_INCLUDE_DIR}
                                     INTERFACE /I${LIBCXX_GENERATED_INCLUDE_TARGET_DIR})
else()
  target_compile_options(cxx-headers INTERFACE -I${LIBCXX_GENERATED_INCLUDE_DIR}
                                     INTERFACE -I${LIBCXX_GENERATED_INCLUDE_TARGET_DIR})
endif()

if (LIBCXX_INSTALL_HEADERS)
  foreach(file ${files})
    get_filename_component(dir ${file} DIRECTORY)
    install(FILES ${file}
      DESTINATION ${LIBCXX_INSTALL_INCLUDE_DIR}/${dir}
      COMPONENT cxx-headers
      PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    )
  endforeach()

  # Install the generated __config_site.
  install(FILES ${LIBCXX_GENERATED_INCLUDE_TARGET_DIR}/__config_site
    DESTINATION ${LIBCXX_INSTALL_INCLUDE_TARGET_DIR}
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    COMPONENT cxx-headers)

  if (NOT CMAKE_CONFIGURATION_TYPES)
    add_custom_target(install-cxx-headers
                      DEPENDS cxx-headers
                      COMMAND "${CMAKE_COMMAND}"
                              -DCMAKE_INSTALL_COMPONENT=cxx-headers
                              -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
    # Stripping is a no-op for headers
    add_custom_target(install-cxx-headers-stripped DEPENDS install-cxx-headers)
  endif()
endif()
