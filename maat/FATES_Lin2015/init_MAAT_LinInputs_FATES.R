dat = read.csv("../Lin2015_cleaned_BDTT.csv",strip.white=T)

# ********************************** STATIC Options **********************************

# define static FUNCTIONS
fnames.static <- list(
  gstar           = 'f_gstar_constref',
  vcmax           = 'f_vcmax_constant',
  jmax           = 'f_jmax_lin',
  tpu           = 'f_tpu_lin',   
  tcor_asc = list(  
    vcmax   = 'f_tcor_asc_Arrhenius',
    jmax    = 'f_tcor_asc_Arrhenius',
    tpu     = 'f_tcor_asc_Arrhenius',
    gstar   = 'f_tcor_asc_Arrhenius',
    Kc      = 'f_tcor_asc_Arrhenius',
    Ko      = 'f_tcor_asc_Arrhenius',
    rd      = 'f_tcor_asc_Arrhenius'
  ),
  tcor_des = list(  
    vcmax   = 'f_tcor_des_modArrhenius',
    jmax    = 'f_tcor_des_modArrhenius',
    tpu   = 'f_tcor_des_modArrhenius',
    rd    = 'f_tcor_des_modArrhenius'
  ),
  tcor_dep = list(  
    tpu   = 'f_tcor_dep_independent',
    rd    = 'f_tcor_dep_independent'
  ),
  deltaS = list(  
    vcmax   = 'f_deltaS_constant',
    jmax    = 'f_deltaS_constant',
    tpu     = 'f_deltaS_constant',
    rd      = 'f_deltaS_constant'
  ),
  etrans    = 'f_etrans_farquharwong1984', 
  Acg       = 'f_Acg_farquhar1980',
  Ajg       = 'f_Ajg_generic', 
  Apg       = 'f_Apg_vonc2000',
  gas_diff  = 'f_gas_diff_ficks_ci', 
  rd        = 'f_rd_lin_N', # <!-- calculate Rd at <reftemp.rd>, f(Vcmax), f(N), -->
  rl_rd     = 'f_rl_rd_fixed',
  ri        = 'f_r_zero',
  rs        = 'f_rs_ball1987', 
  rb        = 'f_r_zero',
  solver    = 'f_solver_brent', 
  residual_func     = 'f_residual_func_leaf_Ar',
  Alim      = 'f_Alim_collatz1991'
) 

# define static Parameters
pars.static <- list(
  a = 0.85,
  f = 0.15,
  ko_kc_ratio = 0.21,
  theta_j = 0.7,
  theta_col_cj = 0.999, 
  theta_col_cjp = 0.999, 
  ajv_25 = 0,
  bjv_25 = 1.67,
  atv_25 = 0,
  btv_25 = 0.16667,
  Apg_alpha = 0,
  g0 = 0.01,
  g1_ball = 8,
  a_rdn_25 = 0.5,
  b_rdn_25 = 0.15,  
  rl_rd_ratio = 1,
  rl_rd_lloyd_a = 0.5,
  rl_rd_lloyd_b = 0.05,
  a_rdv_25_t = 0.015,
  b_rdv_25_t = -0.0005,
  reftemp = list(
    rd = 25,
    vcmax = 25,
    jmax = 25,
    tpu = 25,
    Kc = 25,
    Ko = 25,
    gstar = 25
     ),
  atref = list(
    rd = 0.716,
    vcmax = 50,
    Kc = 40.49,
    Ko = 27.84,
    gstar = 4.275
     ),
  Ha = list(
    vcmax = 65330,
    jmax = 43540,
    rd = 46390,
    tpu = 53100,
    Kc = 79430,
    Ko = 36380,
    gstar = 37830
     ), 
  q10 = list(
    rd = 2
     ), 
  Hd = list(
    vcmax = 149250,
    jmax = 152040,
    tpu = 150650,
    rd = 150650
     ), 
  deltaS = list(
    vcmax = 485,
    jmax = 495,
    tpu = 490,
    rd = 490
     ),  
  R = 8.31446
)

# define static ENVIRONMENTAL VARIABLES
env.static <- list(
  o2_conc = 0.21
)  


# ********************************** Variable Options **********************************

# define variable FUNCTIONS
fnames.var    <- NULL

# define variable PARAMETERS
# pars.var      <- list(
#   g0 = dat[c(1:2872),"g0"],
#   g1_medlyn = dat[c(1:2872),"g1_medlyn"]
# )
pars.var    <- NULL
pars_proc.var <- NULL
pars_eval.var <- NULL

# define variable ENVIRONMENTAL VARIABLES
env.var       <- list(
  ca_conc = dat$CO2S,
  par = dat$PARin,
  rh = dat$RH,
  temp = dat$Tleaf
)
# # #env.var <- NULL
# # env.var       <- list(
#   ca_conc = dat[c(1:2872),"CO2S"],
#   par = dat[c(1:170),"PARin"],
#   rh = dat[c(1:100),"RH"]/100,
#   temp = dat[c(1:2872),"Tleaf"]
# #   ca_conc = seq(50,1500,50)
# # )


################################## Combine the functions, parameters, and environment static and variable lists into a single list 

init_static <- list(
  fnames = list( leaf = fnames.static),
  pars   = list( leaf = pars.static),
  env    = list( leaf = env.static)
)
init_dynamic <- list(
  fnames    = list( leaf = fnames.var),
  pars      = list( leaf = pars.var),
  pars_proc = list( leaf = pars_proc.var),
  pars_eval = list( leaf = pars_eval.var),
  env       = list( leaf = env.var)
)
