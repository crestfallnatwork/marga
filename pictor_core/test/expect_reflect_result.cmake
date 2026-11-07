# Runs pictor_reflect and checks its result against expected diagnostics or --dump output.
#
# Required variables:
#   REFLECTOR  - path to the pictor_reflect executable
#   SHADER     - shader to reflect
#   OUTDIR     - output directory
#   EXPECTED   - "|"-separated substrings that must all appear in the output
# Optional variables:
#   EXPECT_FAILURE   - when set, a non-zero exit and no emitted headers are required
#   EXPECTED_ABSENT  - "|"-separated substrings that must not appear in the output

if(NOT DEFINED REFLECTOR OR NOT DEFINED SHADER OR NOT DEFINED OUTDIR OR NOT DEFINED EXPECTED)
    message(FATAL_ERROR "expect_reflect_result.cmake requires REFLECTOR, SHADER, OUTDIR, EXPECTED")
endif()

file(REMOVE_RECURSE "${OUTDIR}")

execute_process(
    COMMAND "${REFLECTOR}" "${SHADER}" --outdir "${OUTDIR}" --dump
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)

set(combined "${output}\n${error}")

if(EXPECT_FAILURE)
    if(result EQUAL 0)
        message(FATAL_ERROR "pictor_reflect unexpectedly succeeded on ${SHADER}:\n${combined}")
    endif()
    get_filename_component(stem "${SHADER}" NAME_WE)
    if(EXISTS "${OUTDIR}/${stem}.shader.h" OR EXISTS "${OUTDIR}/${stem}.slang.h")
        message(FATAL_ERROR "pictor_reflect wrote headers despite failing on ${SHADER}")
    endif()
else()
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "pictor_reflect failed on ${SHADER}:\n${combined}")
    endif()
endif()

string(REPLACE "|" ";" expected_list "${EXPECTED}")
foreach(item IN LISTS expected_list)
    string(FIND "${combined}" "${item}" found)
    if(found EQUAL -1)
        message(FATAL_ERROR "expected '${item}' not found for ${SHADER}:\n${combined}")
    endif()
endforeach()

if(DEFINED EXPECTED_ABSENT AND NOT EXPECTED_ABSENT STREQUAL "")
    string(REPLACE "|" ";" absent_list "${EXPECTED_ABSENT}")
    foreach(item IN LISTS absent_list)
        string(FIND "${combined}" "${item}" found)
        if(NOT found EQUAL -1)
            message(FATAL_ERROR "unexpected '${item}' found for ${SHADER}:\n${combined}")
        endif()
    endforeach()
endif()

if(DEFINED EXPECTED_HEADER AND NOT EXPECTED_HEADER STREQUAL "")
    get_filename_component(stem "${SHADER}" NAME_WE)
    set(header "${OUTDIR}/${stem}.shader.h")
    if(NOT EXISTS "${header}")
        message(FATAL_ERROR "expected generated header ${header}")
    endif()
    file(READ "${header}" header_text)
    string(REPLACE "|" ";" header_list "${EXPECTED_HEADER}")
    foreach(item IN LISTS header_list)
        string(FIND "${header_text}" "${item}" found)
        if(found EQUAL -1)
            message(FATAL_ERROR "expected '${item}' not found in ${header}:\n${header_text}")
        endif()
    endforeach()
endif()
