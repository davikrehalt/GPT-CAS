if(NOT DEFINED CLI)
  message(FATAL_ERROR "CLI executable was not supplied")
endif()

function(check_cli expected)
  execute_process(
    COMMAND "${CLI}" ${ARGN}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
  )
  if(NOT result EQUAL 0)
    message(FATAL_ERROR
      "CLI failed with ${result}\ncommand args: ${ARGN}\nstderr: ${error}")
  endif()
  if(NOT output STREQUAL expected)
    message(FATAL_ERROR
      "unexpected CLI output\ncommand args: ${ARGN}\nexpected: [${expected}]\nactual: [${output}]")
  endif()
  if(NOT error STREQUAL "")
    message(FATAL_ERROR "successful CLI command wrote to stderr: ${error}")
  endif()
endfunction()

function(check_cli_error expected_pattern)
  execute_process(
    COMMAND "${CLI}" ${ARGN}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
  )
  if(NOT result EQUAL 2)
    message(FATAL_ERROR
      "invalid CLI command exited ${result}\ncommand args: ${ARGN}\nstderr: ${error}")
  endif()
  if(NOT output STREQUAL "")
    message(FATAL_ERROR
      "invalid CLI command produced partial stdout: ${output}")
  endif()
  if(NOT error MATCHES "${expected_pattern}")
    message(FATAL_ERROR
      "invalid CLI diagnostic did not match [${expected_pattern}]: ${error}")
  endif()
endfunction()

function(check_cli_resource expected)
  execute_process(
    COMMAND "${CLI}" ${ARGN}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
  )
  if(NOT result EQUAL 3)
    message(FATAL_ERROR
      "resource-limited CLI command exited ${result}\ncommand args: ${ARGN}\nstderr: ${error}")
  endif()
  if(NOT output STREQUAL expected)
    message(FATAL_ERROR
      "unexpected resource-limited CLI output\ncommand args: ${ARGN}\nexpected: [${expected}]\nactual: [${output}]")
  endif()
  if(NOT error STREQUAL "")
    message(FATAL_ERROR
      "resource result object should not write to stderr: ${error}")
  endif()
endfunction()

check_cli("x^2 + 1/2*y\n"
  --field QQ --vars x,y print "x^2 + 1/2*y")

check_cli("7*y^6 + 5*x*y^4\n"
  --field "GF(101)" --vars x,y diff --var y "x^3+y^7+x*y^5")

check_cli("x - y^2\ny^3 - 1\n"
  --field QQ --vars x,y --order lex gb "x^2-y" "x*y-1")

check_cli("21/5*x^2*y - 3*x^2\n"
  --field QQ --vars x,y nf "x*y^4+y^5"
  --by "x*y^4-21/5*x^2*y" --by "y^5+3*x^2"
  --by "x^2*y^2" --by "x^3")

check_cli("q[0] = x + y\nq[1] = 1\nr = x + y + 1\n"
  --field QQ --vars x,y --order lex divide
  "x^2*y+x*y^2+y^2" --by "x*y-1" --by "y^2-1")

check_cli("6\n"
  --field QQ --vars x,y dim "x^2" "y^3")

check_cli("1\ny\nx\ny^2\nx*y\nx*y^2\n"
  --field QQ --vars x,y std "x^2" "y^3")

check_cli("x\ny\n"
  --field QQ --vars x,y colon --g x "x^2" "x*y" "y^3")

check_cli("y*z - 1\n"
  --field QQ --vars x,y,z eliminate --elim x "x*y-1" "x-z")

check_cli("status=complete\nconclusive=true\nfinite_quotient=true\nsupported_at_origin=true\nlength_Q=2\nlength_P_mod_J2=4\ng_in_J=true\nderivatives_in_J=false\ncycle_valid=false\nprimitive=true\nannihilator_dimension=0\ncolon_quotient_length=2\ncolon_equals_J=true\nannihilator_zero=true\nfaithful_cycle=false\nideal_gb={x^2}\nideal_square_gb={x^4}\ng_remainder=0\nderivative_remainder[x]=2*x\ncolon_gb={x^2}\nideal_in_colon_remainders={0}\ncolon_in_ideal_remainders={0}\n"
  --field QQ --vars x audit --g "x^2" "x^2")

check_cli("status=complete\nconclusive=true\nfinite_quotient=true\nsupported_at_origin=true\nlength_Q=5\nlength_P_mod_J2=10\ng_in_J=true\nderivatives_in_J=true\ncycle_valid=true\nprimitive=true\nannihilator_dimension=0\ncolon_quotient_length=5\ncolon_equals_J=true\nannihilator_zero=true\nfaithful_cycle=true\nideal_gb={x^5}\nideal_square_gb={x^10}\ng_remainder=0\nderivative_remainder[x]=0\ncolon_gb={x^5}\nideal_in_colon_remainders={0}\ncolon_in_ideal_remainders={0}\n"
  --field "GF(5)" --vars x audit --g "x^5" "x^5")

check_cli("status=proper-fixed-point\nconclusive=true\nfaithful_fixed_point=true\nsteps=2\ntransitions=1\nlength_chain=5->5\nstep[0].gb={x^5}\nstep[1].gb={x^5}\ntransition[0].current_subset_next=true\ntransition[0].equal=true\ntransition[0].current_in_next_remainders={0}\ntransition[0].next_in_current_remainders={0}\n"
  --field "GF(5)" --vars x closure --g "x^5" "x^5")

check_cli("status=complete\nconclusive=true\nlength_Q=5\nlength_P_mod_J2=10\nconormal_dimension=5\nh1_dimension=5\nsocle_dimension=1\ncommon_product_space_dimension=1\ncommon_product_space_rank_bound=1\nindividual_rank_lower=1\nindividual_rank_upper=1\nindividual_rank_exact=1\nindividual_rank_proof=proven-full-column-rank\nfull_socle_rank_witness=true\nbest_h1_witness=x^5\nindependent_faithful_colon_replay=true\nfaithful_witness_replay_status=complete\nfaithful_witness_replay_conclusive=true\nfaithful_witness_replay_resource_detail=n/a\ncommon_annihilator_diagnostic_only=true\ncommon_annihilator_diagnostic_gb={x^5}\nconormal_basis={x^5; x^6; x^7; x^8; x^9}\nh1_basis={x^5; x^6; x^7; x^8; x^9}\nsocle_basis={x^4}\naction_matrix_count=5\naction_matrix[0].shape=5x1\naction_matrix[0].entries={{0}; {0}; {0}; {0}; {1}}\naction_matrix[1].shape=5x1\naction_matrix[1].entries={{0}; {0}; {0}; {0}; {0}}\naction_matrix[2].shape=5x1\naction_matrix[2].entries={{0}; {0}; {0}; {0}; {0}}\naction_matrix[3].shape=5x1\naction_matrix[3].entries={{0}; {0}; {0}; {0}; {0}}\naction_matrix[4].shape=5x1\naction_matrix[4].entries={{0}; {0}; {0}; {0}; {0}}\n"
  --field "GF(5)" --vars x h1 "x^5")

check_cli("status=complete\nlength_Q=5\nlength_P_mod_J2=10\ng_in_J=true\nderivatives_in_J=true\ncycle_valid=true\nmultiplication_rank=5\nfull_rank_candidate=true\ncertified_faithful=true\n"
  --field "GF(5)" --vars x screen-audit --g "x^5" "x^5")

check_cli("status=complete\nmaximal_power=5\nlength_Q=5\nlength_P_mod_J2=10\nconormal_dimension=5\nh1_dimension=5\nreduction_shape=5x10\nreduction_nnz=5\nderivative_shape=5x10\nderivative_nnz=4\ncycle_shape=10x10\ncycle_nnz=9\n"
  --field "GF(5)" --vars x cotangent-h1 --maximal-power 5)

check_cli("status=complete\ncycle_valid=true\nmultiplication_rank=5\nannihilator_dimension=0\nfaithful=true\ncolon_equals_J=true\n"
  --field "GF(5)" --vars x verify-h1-class
  --maximal-power 5 --g "x^5")

check_cli("status=not-cycle\ncycle_valid=false\nmultiplication_rank=n/a\nannihilator_dimension=n/a\nfaithful=false\ncolon_equals_J=false\n"
  --field QQ --vars x verify-h1-class --maximal-power 2 --g "x^2")

check_cli("status=complete\nconclusive=true\nlength_Q=5\nlength_P_mod_J2=10\nconormal_dimension=5\nh1_dimension=5\nsocle_dimension=1\nindividual_rank_lower=1\nindividual_rank_upper=1\nindividual_rank_proof=proven-full-column-rank\nfull_rank_candidate=true\ncertified_faithful=true\nwitness=x^5\n"
  --field "GF(5)" --vars x screen-h1 "x^5")

check_cli("status=complete\napolarity=ordinary\nmaximum_dual_degree=2\naction_shape=6x10\naction_nonzeros=5\naction_rank=3\nkernel_dimension=7\ntruncated_kernel_generators=7\nquotient_length=3\nannihilator_gb={x^2 - 2*y; x*y; y^2}\n"
  --field QQ --vars x,y inverse-system "x^2+y")

check_cli("{\"schema\":\"laughable-jg-v1\",\"field\":{\"kind\":\"GF\",\"modulus\":\"5\"},\"variables\":[\"x\"],\"order\":\"grevlex\",\"ideal_generators\":[[{\"coefficient\":\"1\",\"exponents\":[5]}]],\"g\":[{\"coefficient\":\"1\",\"exponents\":[5]}]}\n"
  --field "GF(5)" --vars x certificate --g "x^5" "x^5")

check_cli_resource("status=resource-limit\nconclusive=false\nfaithful_fixed_point=false\nsteps=1\ntransitions=0\nlength_chain=5\nstep[0].gb={x^5}\nresource_detail=colon closure reached its maximum number of steps\n"
  --field "GF(5)" --vars x closure --g "x^5" --max-steps 0 "x^5")

execute_process(
  COMMAND "${CLI}" --help
  RESULT_VARIABLE help_result
  OUTPUT_VARIABLE help_output
  ERROR_VARIABLE help_error
)
if(NOT help_result EQUAL 0 OR NOT help_error STREQUAL "")
  message(FATAL_ERROR
    "--help failed: result=${help_result}, stderr=${help_error}")
endif()
if(NOT help_output MATCHES
    "3  computation was inconclusive because of a resource limit")
  message(FATAL_ERROR
    "--help does not document resource exit 3: ${help_output}")
endif()

check_cli_error("--g is valid only"
  --field QQ --vars x print x --g x)

check_cli_error("requires --g"
  --field QQ --vars x colon "x^2")

check_cli_error("unknown variable"
  --field QQ --vars x eliminate --elim y "x^2")

check_cli_error("--max-steps is valid only"
  --field QQ --vars x audit --g "x^2" --max-steps 1 "x^2")

check_cli_error("finite-field discovery command"
  --field QQ --vars x screen-h1 "x^2")

check_cli_error("--apolarity must be"
  --field QQ --vars x inverse-system --apolarity nope "x^2")

check_cli_error("nonnegative decimal integer"
  --field QQ --vars x closure --g "x^2" --max-steps nope "x^2")

check_cli_error("requires --maximal-power"
  --field QQ --vars x cotangent-h1)

check_cli_error("positive decimal integer"
  --field QQ --vars x cotangent-h1 --maximal-power 0)

check_cli_error("--maximal-power is valid only"
  --field QQ --vars x print --maximal-power 2 x)

execute_process(
  COMMAND "${CLI}" --field QQ --vars x print "2x"
  RESULT_VARIABLE bad_result
  OUTPUT_VARIABLE bad_output
  ERROR_VARIABLE bad_error
)
if(NOT bad_result EQUAL 2)
  message(FATAL_ERROR "invalid expression should exit 2, got ${bad_result}")
endif()
if(NOT bad_output STREQUAL "")
  message(FATAL_ERROR "invalid expression produced partial stdout: ${bad_output}")
endif()
if(NOT bad_error MATCHES "parse error")
  message(FATAL_ERROR "invalid expression lacks parse diagnostic: ${bad_error}")
endif()

execute_process(
  COMMAND "${CLI}" --field QQ --vars x print x --by x
  RESULT_VARIABLE ignored_option_result
  OUTPUT_VARIABLE ignored_option_output
  ERROR_VARIABLE ignored_option_error
)
if(NOT ignored_option_result EQUAL 2)
  message(FATAL_ERROR
    "command-inapplicable option should exit 2, got ${ignored_option_result}")
endif()
if(NOT ignored_option_output STREQUAL "")
  message(FATAL_ERROR
    "command-inapplicable option produced stdout: ${ignored_option_output}")
endif()
if(NOT ignored_option_error MATCHES "valid only")
  message(FATAL_ERROR
    "command-inapplicable option lacks diagnostic: ${ignored_option_error}")
endif()

execute_process(
  COMMAND "${CLI}" --field QQ --vars x print x --var=
  RESULT_VARIABLE empty_option_result
  OUTPUT_VARIABLE empty_option_output
  ERROR_VARIABLE empty_option_error
)
if(NOT empty_option_result EQUAL 2 OR NOT empty_option_output STREQUAL "")
  message(FATAL_ERROR
    "explicit empty inapplicable option was not rejected: result=${empty_option_result}, output=${empty_option_output}")
endif()
