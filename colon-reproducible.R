rm(list=ls())
library(flexsurv)
library(splines)
library(survival)
library(rstpm2)
library(pec)
library(riskRegression)
library(randomForestSRC)
library(simsurv)
library(mboost)
library(timereg)
library(mltools)
library(data.table)

library(tictoc) # for timing

## for Neural Networks see "Neural Networks for Survival Analysis in R" by Raphael Sonabend
## https://sebastian.vollmer.ms/post/survival_networks/

# options(repos=c(
#   mlrorg = 'https://mlr-org.r-universe.dev',
#   raphaels1 = 'https://raphaels1.r-universe.dev',
#   CRAN = 'https://cloud.r-project.org'
# ))
# install.packages(c("ggplot2", "mlr3benchmark", "mlr3pipelines", "mlr3proba", "mlr3tuning", 
#                    "survivalmodels", "mlr3extralearners"))

library(survivalmodels)

install_pycox(pip = TRUE, install_torch = TRUE)
install_keras(pip = TRUE, install_tensorflow = TRUE)

library(mlr3)
library(mlr3proba)
library(paradox)
library(mlr3tuning)
library(mlr3extralearners)

## ---------------------------------------------------------------------------
## Functions to predict and use the Score function of riskregression package
## ---------------------------------------------------------------------------

## Prediction for flexible Regression 
predictRisk.stpm2 <- function(object, newdata, times, ...){
  ntempi <- length(times)
  pred <- NULL
  for(i in 1:ntempi){
    newdata$eventtime <- times[i]
    pred0 <- 1-predict(object, newdata)
    pred <- cbind(pred, pred0)
    pred
  }
  return(pred)
}

## predictions for boosting
predictRisk.mboost <- function(object, newdata, times, ...){
  prova <- survFit(object, newdata)
  pred <- NULL
  for(i in times){
    pred0 <- 1-prova$surv[which.min(abs( prova$time - i)),]
    pred <- cbind(pred, pred0)
    pred
  }
  return(pred)
}

## predictions for random forests
predictRisk.rfsrc <- function(object, newdata, times, ...){
  prova <- predict(object, newdata)
  pred <- NULL
  for(i in times){
    pred0 <- 1-prova$survival[,which.min(abs( prova$time.interest - i))]
    pred <- cbind(pred, pred0)
    pred
  }
  return(pred)
}

## prediction for Neural networks in mlr3
predictRisk.Learner <- function(object, newdata, times, cent1, cent2, sca1, sca2){
  newdata2 <- newdata
  newdata2$age <- scale(newdata2$age, center=cent1, scale = sca1)
  newdata2$nodes <- scale(newdata2$nodes, center=cent2, scale = sca2)
  temp <- one_hot(as.data.table(newdata2$rx) )
  newdata2$Obs <- as.numeric(temp$V1_Obs)
  newdata2$Lev <- as.numeric(temp$V1_Lev)
  newdata2$LFU <- as.numeric(temp$`V1_Lev+5FU`)
  temp <- one_hot(as.data.table(newdata2$differ) )
  newdata2$differ1 <- temp$V1_1
  newdata2$differ2 <- temp$V1_2
  newdata2$differ3 <- temp$V1_3
  head(newdata2)
  newdata3 <- as.data.frame(newdata2[, -c(1, 8, 12)])
  ntempi <- length(times)
  pred <- NULL
  temp <- object$predict_newdata(newdata3)
  for(i in times){
    pred0 <- as.numeric(temp$distr$cdf(i))
    pred <- cbind(pred, pred0)
    pred
  }
  return(pred)
}



# -------------------------------------------------------------------------------------------
####                          colon Cancer data
# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
# The colon data are available in R in the package survival. 
# These are data from a trial of Levamisole vs Levamisole + 5-FU 
# chemotherapy agent. There are two records per person, one for recurrence and one for death.
# ------------------------------------------------------------------------------------------

# etype:	event type: 1=recurrence,2=death
colon2 <- subset(survival::colon, subset=etype==1)
colon2$months <- colon2$time/30.4167

colon2$inv.serosa <- 1*(colon2$extent %in% c(3,4))
colon2$differ <- as.factor(colon2$differ)

##  ----------------------------------------------------------------------------
####                       Riley's Sample Size
##  ----------------------------------------------------------------------------
library(pmsampsize)
## the following calculations are described in supplementary material (S5) of
# Riley R D, Ensor J, Snell K I E, Harrell F E, Martin G P, Reitsma J B et al. 
# Calculating the sample size required for developing a clinical prediction model 
# BMJ 2020; 368 :m441 doi:10.1136/bmj.m441

# estimate the monthly event rate
 colon2$months <- colon2$time/30.4167
 exColon <- flexsurvreg(Surv(months, status) ~ 1, data=colon2, dist="exponential")
 exColon
 
 mfup <- survfit(Surv(months, status==0) ~ 1, data=colon2)
 mfup
 print(mfup, print.rmean=TRUE, rmean=72)
 
 lambda = 68*exColon$res[1]
 lambda
 
 lnNULL <- lambda * 100 * log(lambda) - lambda * 100
 
 maxR <- 1 - exp(2*lnNULL / 100)
 maxR
 
 pmsampsize(type = "s", rsquared = .1*maxR, parameters = 23, rate = exColon$res[1],
            timepoint = 72, meanfup = 68)
 
 # NB: Assuming 0.05 acceptable difference in apparent & adjusted R-squared 
 # NB: Assuming 0.05 margin of error in estimation of overall risk at time point = 72  
 # NB: Events per Predictor Parameter (EPP) assumes overall event rate = 0.01090496  
 # 
 # Samp_size Shrinkage Parameter        Rsq Max_Rsq   EPP
 # Criteria 1        2306     0.900        23 0.08543513    0.85 74.35 # small overfitting defined by an expected shrinkage of predictor effects by 10% or less,
 # Criteria 2         507     0.668        23 0.08543513    0.85 16.35 # small absolute difference of 0.05 in the model's apparent and adjusted Nagelkerke's R-squared value
 # Criteria 3 *      2306     0.900        23 0.08543513    0.85 74.35 # precise estimation (within +/- 0.05) of the average outcome risk in the population for a key timepoint of interest for prediction.
 # Final             2306     0.900        23 0.08543513    0.85 74.35
 # 
 # Minimum sample size required for new model development based on user inputs = 2306, 
 # corresponding to 156808 person-time** of follow-up, with 1710 outcome events 
 # assuming an overall event rate = 0.01090496 and therefore an EPP = 74.35  
 # 
 # * 95% CI for overall risk = (0.527, 0.561), for true value of 0.544 and sample size n = 2306 
 # **where time is in the units mean follow-up time was specified in
 
 ##  ----------------------------------------------------------------------------
 # Creation of training set and validation/test set: section 2.1
 # - We randomly divided the original dataset into a training set (3/4 of the patients) 
 # and a validation set for performance assessment. 
 ##  ----------------------------------------------------------------------------
 
colon3 <- na.omit(colon2)

set.seed(1)
n <- nrow(colon3)
n.left = round(3*n/4)
n.left
split <- sample.int(n, size = n.left)
colon3$id2 <- 1:n
colon3$train <- colon3$id2 %in% split

colon2.exp <- colon3[rep(seq(nrow(colon3)),20),]

##  ----------------------------------------------------------------------------
## generate data from this flexible model with non linear effects and 
## time dependent effects
## -----------------------------------------------------------------------------
## for truncated power cubic splines code from:
## Perperoglou, A., Sauerbrei, W., Abrahamowicz, M. et al. 
## A review of spline function procedures in R. BMC Med Res Methodol 19, 46 (2019). 
## https://doi.org/10.1186/s12874-019-0666-3
## Additional file 1: Appendix R code.

## Code 1: truncated power splines
D <- 3 # degree of series
knots <- 50 # knot
age.sp <- outer (colon2.exp$age, knots,">") * outer (colon2.exp$age, knots, "-")^D 
colon2.exp$age2 <- age.sp[,1]

knots <- 2 # knot
nodes.sp <- outer (colon2.exp$nodes, knots,">") * outer (colon2.exp$nodes, knots, "-")^D 
colon2.exp$nodes2 <- nodes.sp[,1]

##  ----------------------------------------------------------------------------
## generate data from a flexible model with non linear effects and interactions
## -----------------------------------------------------------------------------

true_mod <- stpm2(Surv(time, status)~(age + age2) * rx + sex + obstruct + perfor + adhere + 
                    nodes + nodes2 + node4 + 
                    differ + inv.serosa + surg, data=colon2.exp,
                    smooth.formula=~nsx(log(time), df=3))
summary(true_mod)
attr(true_mod@termsd, "predvars")
# list(logHhat, age, age2, rx, sex, obstruct, perfor, adhere, nodes, 
#      nodes2, node4, differ, inv.serosa, surg, nsx(log(time), knots = c(`33.33333%` = 5.56068163101553, 
#                                                                        `66.66667%` = 6.35957386867238), 
#                                                   Boundary.knots = c(2.07944154167984, 
#                                                                      7.8991534833431), 
#                                                   intercept = FALSE, derivs = c(2, 2), centre = FALSE, 
#                                                   log = FALSE))
# Define a function returning the log cum hazard at time t

# ----------------------------------------------------------------
# see simsurv package vignette:
# https://cran.r-project.org/web/packages/simsurv/vignettes/simsurv_usage.html
# example 2
# ----------------------------------------------------------------
# Define a function returning the log cum hazard at time t
# ----------------------------------------------------------------

logcumhaz <- function(t, x, betas){
  
  # Obtain the basis terms for the spline-based log
  # cumulative hazard (evaluated at time t)

    basis <- rstpm2::nsx(log(t), knots = c(`33.33333%` = 5.56068163101553, 
                                         `66.66667%` = 6.35957386867238), 
                               Boundary.knots = c(2.07944154167984, 
                                                  7.8991534833431), 
                       intercept = FALSE, derivs = c(2, 2), centre = FALSE, 
                       log = FALSE)
  
  # Evaluate the log cumulative hazard under the
  # Royston and Parmar specification
  res <- 
    betas[["(Intercept)"]] +
    betas[["nsx(log(time), df = 3)1"]] * basis[,1] + 
    betas[["nsx(log(time), df = 3)2"]] * basis[,2] +
    betas[["nsx(log(time), df = 3)3"]] * basis[,3] +
    betas[["age"]] * x[["age"]] +
    betas[["age2"]] * x[["age2"]] +
    betas[["rxLev"]] * x[["rxLev"]] +
    betas[["rxLev+5FU"]] * x[["rxLev+5FU"]] + 
    betas[["sex"]] * x[["sex"]] +
    betas[["obstruct"]] * x[["obstruct"]] +
    betas[["perfor"]] * x[["perfor"]] +
    betas[["adhere"]] * x[["adhere"]] + 
    betas[["nodes"]] * x[["nodes"]] + 
    betas[["nodes2"]] * x[["nodes2"]] + 
    betas[["node4"]] * x[["node4"]] +
    betas[["differ2"]] * x[["differ2"]] +
    betas[["differ3"]] * x[["differ3"]] +
    betas[["inv.serosa"]] * x[["inv.serosa"]] +
    betas[["surg"]] * x[["surg"]] +
    betas[["age:rxLev"]] * x[["age:rxLev"]] +
    betas[["age:rxLev+5FU"]] * x[["age:rxLev+5FU"]] +
    betas[["age2:rxLev"]] * x[["age2:rxLev"]] +
    betas[["age2:rxLev+5FU"]] * x[["age2:rxLev+5FU"]] 
    
  # Return the log cumulative hazard at time t
  res
}



gen_data <- function(true_mod) {
  # Create a data frame with the subject IDs and treatment covariate
  cov <- as.data.frame(true_mod@x)
  # Simulate the event times
  dat <- simsurv(betas = coef(true_mod),    # "true" parameter values
                 x = cov,                   # covariate data 
                 logcumhazard = logcumhaz,  # definition of log cum hazard
                 maxt = 3650,               # administrative right-censoring
                 interval = c(1E-8,1E5))    # interval for root finding
  table(dat$status)/nrow(dat)

  dat <- cbind(colon2.exp[,c(3:9, 11, 13:14, 18, 20:22)], dat[, c(2:3)])
  dat <- as.data.frame(dat)
  dat$train <- colon2.exp$train
  table(dat$train)
  return(dat)
}

##  ----------------------------------------------------------------------------
#             Main function for simulation
##  ----------------------------------------------------------------------------
#  This function is for illustrative purposes. The true simulation was run 
#  fitting separately the different models considered the amount of time required.
#  simulations were run in parallel using HPC.
#  Computational resources provided by INDACO Platform, 
#  which is a project of High Performance Computing at the University of MILAN 
#  https://www.indaco.unimi.it/
##  ----------------------------------------------------------------------------

sim_run <- function(true_mod) {
  
  tic("data generation")
  dat <- gen_data(true_mod)
  toc()
  
  ## training data
  dataleft.tot <-dat[dat$train,]
  dim(dataleft.tot)
  ## sample training data
  sample.train <- sample.int(nrow(dataleft.tot), size = 1000)    
  dataleft <- dataleft.tot[sample.train,]
  ## validatation/test data
  dataright <- dat[!dat$train,]
  
  ## different models for comparing predictions
  
  tic("Cox and AFT models ")
  mod.cox <- coxph(Surv(eventtime, status) ~ age + rx + sex + obstruct + perfor + adhere + 
                     nodes + nodes2 + node4 + differ + inv.serosa + surg, data = dataleft, x=TRUE)
  
  mod.cox.nl <- coxph(Surv(eventtime, status) ~ ns(age,3) + rx + sex + obstruct + perfor + adhere + 
                      ns(nodes,3) + nodes2 + node4 + differ + inv.serosa + surg, data = dataleft, x=TRUE)

  mod.cox.nl.int <- coxph(Surv(eventtime, status) ~ ns(age,3) * rx + sex + obstruct + perfor + adhere + 
                          ns(nodes,3) + nodes2 + node4 + differ + inv.serosa + surg, data = dataleft, x=TRUE)

  # true model
  D <- 3 # degree of series
  knots <- c(50) # number of knots
  age.sp <- outer (dataleft$age, knots,">") * outer (dataleft$age, knots, "-")^D # as in eq.3
  dataleft$age2 <- age.sp[,1]
  knots <- 2 # number of knots
  nodes.sp <- outer (dataleft$nodes, knots,">") * outer (dataleft$nodes, knots, "-")^D # as in eq.3
  dataleft$nodes2 <- nodes.sp[,1]
  
  mod.stpm <- stpm2(Surv(eventtime, status)~(age + age2) * rx + sex + obstruct + perfor + adhere + 
                      nodes + nodes2 + node4 + 
                      differ + inv.serosa + surg, data=dataleft,
                    smooth.formula=~nsx(log(eventtime), df=3))

  tic("Cox and flexible models estimated ... ")
  toc()
  
  ## Boosting
  tic("Boosting Cox and AFT models")
  mod.mb <- gamboost(Surv(eventtime, status) ~ bbs(age) + bols(rx) + bols(sex) +
                     bols(obstruct) + bols(perfor) + bols(adhere) +
                     bbs(nodes) + bols(node4) + bols(inv.serosa) + bols(surg), data = dataleft, family=CoxPH(),
                     control = boost_control(mstop = 1000, center = TRUE))

    cv10f <- cv(model.weights(mod.mb), type = "kfold")
    cvm <- cvrisk(mod.mb, folds = cv10f, papply = lapply, grid=seq(50, 10050, 100))
    mod.mb[mstop(cvm)]
  toc()
    ## Random Forest SRC

    tic("Random Forest SRC")
    
    TUNING <- randomForestSRC::tune(Surv(eventtime, status) ~ age + rx + sex + obstruct + perfor + adhere + 
                                      nodes + nodes2 + node4 + differ + inv.serosa + surg, data = dataleft, ntreeTry = 100)
    mod.rf <- rfsrc.fast(Surv(eventtime, status) ~ age + rx + sex + obstruct + perfor + adhere + 
                           nodes + nodes2 + node4 + differ + inv.serosa + surg, data = dataleft, mtry=TUNING$optimal[2],
                         nodesize = TUNING$optimal[1], ntree = 5000, forest = T)

    toc()
    ## data preparation (scaling and one-hot encoding) for Neural Network
    ## Training data
    
    tic("PC hazard DNN")
    
    dataleft2 <- dataleft
    dataleft2$age <- scale(dataleft2$age)
    dataleft2$nodes <- scale(dataleft2$nodes)
    temp <- one_hot(as.data.table(dataleft2$rx) )
    dataleft2$Obs <- temp$V1_Obs
    dataleft2$Lev <- temp$V1_Lev
    dataleft2$LFU <- temp$`V1_Lev+5FU`
    temp <- one_hot(as.data.table(dataleft2$differ) )
    dataleft2$differ1 <- temp$V1_1
    dataleft2$differ2 <- temp$V1_2
    dataleft2$differ3 <- temp$V1_3
    head(dataleft2)
    dataleft3 <- dataleft2[, -c(1, 8, 12)]
    head(dataleft3)
    
    CENTER=c(
      attr(dataleft2$age, "scaled:center"),
      attr(dataleft2$nodes, "scaled:center"))
    SCALE=c(
      attr(dataleft2$age, "scaled:scale"),
      attr(dataleft2$nodes, "scaled:scale"))
  
    task_colon = TaskSurv$new(id = "dataleft3", backend = dataleft3,
                              time = "eventtime", event = "status")
    
    search_space <- ps(
      ## p_dbl for numeric valued parameters
      dropout = p_dbl(lower = 0, upper = .1),
      weight_decay = p_dbl(lower = 0, upper = .1),
      learning_rate = p_dbl(lower = 0, upper = .1),
      ## p_int for integer valued parameters
      nodes = p_int(lower = 64, upper = 256),
      k = p_int(lower = 1, upper = 8)
    )
    
    search_space$trafo <- function(x, param_set) {
      x$num_nodes = rep(x$nodes, x$k)
      x$nodes = x$k = NULL
      return(x)
    }
    
    ## load learners
    learner.dh <- lrn("surv.pchazard",
                      frac = 0.3, early_stopping = TRUE, epochs = 10000,
                      optimizer = "adam"#, learning_rate = 0.1, weight_decay = 0.0001
    )
    
    learner.dh$encapsulate = c(train = "evaluate")
    learner.dh$fallback = lrn("surv.coxph")
    
    at <- AutoTuner$new(
      learner = learner.dh,
      search_space = search_space,
      resampling = rsmp("cv", folds = 3),
      measure = msr("surv.cindex"),
      terminator = trm("evals", n_evals = 10), # n_evals = 250
      tuner = tnr("random_search")
    )
    
    at$train(task_colon)
    print("NN estimated ... ")
    at$tuning_result
    toc()
    
  tic("Scoring")
  res <- Score(list("mod.cox.nl"=mod.cox.nl,
                        "mod.cox.nl.int" = mod.cox.nl.int,
                        "mod.cox"=mod.cox, 
                        "mod.stpm"=mod.stpm,
                        "mod.mb" = mod.mb,
                        "mod.rf.fast"=mod.rf,
                        "nn" = at
  ),
  formula=Surv(eventtime,status)~1,
  data=dataright,
  conf.int=FALSE, times=c(1460, 1825, 2190),
  summary=c("risks","IPA", "ibs"),
  predictRisk.args = list(Learner = list(CENTER[1], CENTER[2],
                                         SCALE[1],  SCALE[2]))
  )
  toc()

  return(list(res = res))
}

res <- sim_run(true_mod)

res$res$Brier$score



