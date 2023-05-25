rm(list=ls())
library(condSURV)
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

## for Neural Networks see "Neural Networks for Survival Analysis in R" by Raphael Sonabend
## https://sebastian.vollmer.ms/post/survival_networks/

options(repos=c(
  mlrorg = 'https://mlr-org.r-universe.dev',
  raphaels1 = 'https://raphaels1.r-universe.dev',
  CRAN = 'https://cloud.r-project.org'
))
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
predictRisk.Learner <- function(object, newdata, times, cent1, cent2, cent3, cent4, cent5, 
                                sca1, sca2, sca3, sca4, sca5){
  newdata2 <- newdata
  newdata2$age <- scale(newdata2$age, center=cent1, scale = sca1)
  newdata2$menopause <- newdata2$menopause-1
  newdata2$hormone <- newdata2$hormone-1
  newdata2$size <- scale(newdata2$size, center=cent2, scale = sca2)
  newdata2$grade <- as.factor(newdata2$grade)
  temp <- one_hot(as.data.table(newdata2$grade) )
  newdata2$G1 <- temp$V1_1
  newdata2$G2 <- temp$V1_2
  newdata2$G3 <- temp$V1_3
  newdata2$nodes <- scale(newdata2$nodes, center=cent3, scale = sca3)
  newdata2$prog_recp <- scale(newdata2$prog_recp, center=cent4, scale = sca4)
  newdata2$estrg_recp <- scale(newdata2$estrg_recp, center=cent5, scale = sca5)
  newdata3 <- newdata2[, -c(5, 9, 12)]
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

##  ----------------------------------------------------------------------------
####     German Breast Cancer data
##  ----------------------------------------------------------------------------

gbcsCS$months <- gbcsCS$survtime/30.4167
gbcsCS$grade <- as.factor(gbcsCS$grade)

table(gbcsCS$censdead)/nrow(gbcsCS)

plot(Surv(survtime, censdead) ~ 1, data=gbcsCS)

##  ----------------------------------------------------------------------------
####                       Riley's Sample Size
##  ----------------------------------------------------------------------------
library(pmsampsize)
## the following calculations are described in supplementary material (S5) of
# Riley R D, Ensor J, Snell K I E, Harrell F E, Martin G P, Reitsma J B et al. 
# Calculating the sample size required for developing a clinical prediction model 
# BMJ 2020; 368 :m441 doi:10.1136/bmj.m441

# estimate the monthly event rate
gbcsCS$months <- gbcsCS$survtime/30.4167
exGBCS <- flexsurvreg(Surv(months, censdead) ~ 1, data=gbcsCS, dist="exponential")
exGBCS 

# calculate the log of the exponential likelihood in nstandardized time-units
mfup <- survfit(Surv(months, censdead==0) ~ 1, data=gbcsCS)
mfup
print(mfup, print.rmean=TRUE, rmean=72)
lambda = 50*exGBCS$res[1]
lnNULL <- lambda * 100 * log(lambda) - lambda * 100

# calculate max R2CS:
maxR <- 1 - exp(2*lnNULL / 100)
maxR

# sample size calculation with 21 parameters
pmsampsize(type = "s", rsquared = .1*maxR, parameters = 21, rate = exGBCS$res[1],
           timepoint = 72, meanfup = 50)


# NB: Assuming 0.05 acceptable difference in apparent & adjusted R-squared 
# NB: Assuming 0.05 margin of error in estimation of overall risk at time point = 72  
# NB: Events per Predictor Parameter (EPP) assumes overall event rate = 0.005741281  
# 
#              Samp_size Shrinkage Parameter        Rsq Max_Rsq   EPP
# Criteria 1        2384     0.900        21 0.07590561    0.76 39.11 # small overfitting defined by an expected shrinkage of predictor effects by 10% or less,
# Criteria 2         520     0.666        21 0.07590561    0.76  8.53 # small absolute difference of 0.05 in the model's apparent and adjusted Nagelkerke's R-squared value
# Criteria 3 *      2384     0.900        21 0.07590561    0.76 39.11 # precise estimation (within +/- 0.05) of the average outcome risk in the population for a key timepoint of interest for prediction.
# Final             2384     0.900        21 0.07590561    0.76 39.11
# 
# Minimum sample size required for new model development based on user inputs = 2384, 
# corresponding to 143040 person-time** of follow-up, with 822 outcome events 
# assuming an overall event rate = 0.005741281 and therefore an EPP = 39.11  
# 
# * 95% CI for overall risk = (0.32, 0.357), for true value of 0.339 and sample size n = 2384 
# **where time is in the units mean follow-up time was specified in

# sample size calculation with 30 parameters
pmsampsize(type = "s", rsquared = .1*maxR, parameters = 30, rate = exGBCS$res[1],
           timepoint = 72, meanfup = 60)

# NB: Assuming 0.05 acceptable difference in apparent & adjusted R-squared 
# NB: Assuming 0.05 margin of error in estimation of overall risk at time point = 72  
# NB: Events per Predictor Parameter (EPP) assumes overall event rate = 0.005741281  
# 
# Samp_size Shrinkage Parameter        Rsq Max_Rsq   EPP
# Criteria 1        3405     0.900        30 0.07590561    0.76 39.10
# Criteria 2         743     0.666        30 0.07590561    0.76  8.53
# Criteria 3 *      3405     0.900        30 0.07590561    0.76 39.10
# Final             3405     0.900        30 0.07590561    0.76 39.10
# 
# Minimum sample size required for new model development based on user inputs = 3405, 
# corresponding to 204300 person-time** of follow-up, with 1173 outcome events 
# assuming an overall event rate = 0.005741281 and therefore an EPP = 39.1  
# 
# * 95% CI for overall risk = (0.323, 0.354), for true value of 0.339 and sample size n = 3405 
# **where time is in the units mean follow-up time was specified in


##  ----------------------------------------------------------------------------
# Creation of training set and validation/test set: section 2.1
# - We randomly divided the original dataset into a training set (3/4 of the patients) 
# and a validation set for performance assessment. 
##  ----------------------------------------------------------------------------

n <- nrow(gbcsCS)
n.left = round(3*n/4)
n.left
set.seed(1)
split <- sample.int(n, size = n.left)
gbcsCS$id2 <- 1:n
gbcsCS$train <- gbcsCS$id2 %in% split
table(gbcsCS$train)

##  ----------------------------------------------------------------------------
# - Two "artificial cohorts" were then obtained by separately replicating 20 times the training and validation cohorts. 
##  ----------------------------------------------------------------------------
gbcs.exp <- gbcsCS[rep(seq(nrow(gbcsCS)),20),]
dim(gbcs.exp)
table(gbcs.exp$train)

gbcs.exp$eventtime <- gbcs.exp$survtime
gbcs.exp$status <- gbcs.exp$censdead

##  ----------------------------------------------------------------------------
## generate data from a flexible model with non linear effects, interactions and 
## time dependent effects
## -----------------------------------------------------------------------------
true_mod <- stpm2(Surv(eventtime, status)~nsx(age,3) + menopause + hormone + nsx(size,3) * grade + nodes + 
                    grade + nsx(prog_recp,3) + nsx(estrg_recp, 3) , data=gbcs.exp,
                  smooth.formula=~nsx(log(eventtime), df=3) + nsx(estrg_recp, 3) : log(eventtime))

summary(true_mod)
# True model coefficients

# Maximum likelihood estimation
# 
# Call:
#   stpm2(formula = Surv(eventtime, status) ~ nsx(age, 3) + menopause + 
#           hormone + nsx(size, 3) * grade + nodes + grade + nsx(prog_recp, 
#                                                                3) + nsx(estrg_recp, 3), data = gbcs.exp, smooth.formula = ~nsx(log(eventtime), 
#                                                                                                                                df = 3) + nsx(estrg_recp, 3):log(eventtime))
# 
# Coefficients:
#                                     Estimate  Std. Error  z value     Pr(z)    
# (Intercept)                         -8.4041655   0.6083603 -13.8145 < 2.2e-16 ***
# nsx(age, 3)1                         0.1320267   0.1248282   1.0577  0.290207    
# nsx(age, 3)2                        -0.2684084   0.2699843  -0.9942  0.320144    
# nsx(age, 3)3                         0.2759006   0.1584789   1.7409  0.081696 .  
# menopause                            0.0480045   0.0658285   0.7292  0.465858    
# hormone                             -0.2913718   0.0386602  -7.5367 4.818e-14 ***
# nsx(size, 3)1                        1.8161441   0.4461522   4.0707 4.688e-05 ***
# nsx(size, 3)2                       -2.5530061   1.1984997  -2.1302  0.033158 *  
# nsx(size, 3)3                       -3.2930280   1.2071274  -2.7280  0.006372 ** 
# grade                               -0.5136930   0.2222551  -2.3113  0.020818 *  
# nodes                                0.0585470   0.0023342  25.0823 < 2.2e-16 ***
# nsx(prog_recp, 3)1                  -2.0081080   0.1125280 -17.8454 < 2.2e-16 ***
# nsx(prog_recp, 3)2                  -2.1944888   0.1148285 -19.1110 < 2.2e-16 ***
# nsx(prog_recp, 3)3                  -1.8903803   0.1072588 -17.6245 < 2.2e-16 ***
# nsx(estrg_recp, 3)1                 -1.0910272   1.8138912  -0.6015  0.547517    
# nsx(estrg_recp, 3)2                -28.2985932   2.1482679 -13.1727 < 2.2e-16 ***
# nsx(estrg_recp, 3)3                -50.1935220   4.5732111 -10.9756 < 2.2e-16 ***
# nsx(log(eventtime), df = 3)1         5.8829417   0.2019280  29.1339 < 2.2e-16 ***
# nsx(log(eventtime), df = 3)2        13.9281136   0.6029063  23.1016 < 2.2e-16 ***
# nsx(log(eventtime), df = 3)3         5.1529228   0.1474962  34.9360 < 2.2e-16 ***
# nsx(size, 3)1:grade                 -0.5691928   0.1954886  -2.9116  0.003595 ** 
# nsx(size, 3)2:grade                  2.4348495   0.5533537   4.4002 1.082e-05 ***
# nsx(size, 3)3:grade                  2.3882672   0.5623184   4.2472 2.165e-05 ***
# nsx(estrg_recp, 3)1:log(eventtime)   0.0675550   0.2463643   0.2742  0.783925    
# nsx(estrg_recp, 3)2:log(eventtime)   3.8656070   0.2961978  13.0508 < 2.2e-16 ***
# nsx(estrg_recp, 3)3:log(eventtime)   6.9999201   0.6336899  11.0463 < 2.2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# -2 log L: 61583.28 

attr(true_mod@termsd[9], "predvars")

# list(logHhat, nsx(log(eventtime), knots = c(`33.33333%` = 6.50328867253789, 
#                                             `66.66667%` = 7.05701036485508), 
#                   Boundary.knots = c(4.27666611901606,
#                                      7.80384330353877), 
#                   intercept = FALSE, derivs = c(2, 2), centre = FALSE, 
#                   log = FALSE))

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
  # basis <- flexsurv::basis(knots, log(t))
  basis <- rstpm2::nsx(log(t), knots = c(`33.33333%` = 6.50328867253789, 
                                         `66.66667%` = 7.05701036485508), 
                       Boundary.knots = c(4.27666611901606, 
                                          7.80384330353877), 
                       intercept = FALSE, derivs = c(2, 2), centre = FALSE, 
                       log = FALSE)
  
  # Evaluate the log cumulative hazard under the
  # Royston and Parmar specification
  res <- 
    betas[["(Intercept)"]] +
    betas[["nsx(log(eventtime), df = 3)1"]] * basis[,1] + 
    betas[["nsx(log(eventtime), df = 3)2"]] * basis[,2] +
    betas[["nsx(log(eventtime), df = 3)3"]] * basis[,3] +
    betas[["nsx(age, 3)1"]] * x[["nsx(age, 3)1"]] +
    betas[["nsx(age, 3)2"]] * x[["nsx(age, 3)2"]] +
    betas[["nsx(age, 3)3"]] * x[["nsx(age, 3)3"]] +
    betas[["menopause"]] * x[["menopause"]] +
    betas[["hormone"]] * x[["hormone"]] + 
    betas[["nsx(size, 3)1"]] * x[["nsx(size, 3)1"]] +
    betas[["nsx(size, 3)2"]] * x[["nsx(size, 3)2"]] +
    betas[["nsx(size, 3)3"]] * x[["nsx(size, 3)3"]] +
    betas[["nodes"]] * x[["nodes"]] + 
    betas[["grade2"]] * x[["grade2"]] + 
    betas[["grade3"]] * x[["grade3"]] + 
    betas[["nsx(prog_recp, 3)1"]] * x[["nsx(prog_recp, 3)1"]] +
    betas[["nsx(prog_recp, 3)2"]] * x[["nsx(prog_recp, 3)2"]] +
    betas[["nsx(prog_recp, 3)3"]] * x[["nsx(prog_recp, 3)3"]] +
    betas[["nsx(estrg_recp, 3)1"]] * x[["nsx(estrg_recp, 3)1"]] +
    betas[["nsx(estrg_recp, 3)2"]] * x[["nsx(estrg_recp, 3)2"]] +
    betas[["nsx(estrg_recp, 3)3"]] * x[["nsx(estrg_recp, 3)3"]] +
    betas[["nsx(size, 3)1:grade2"]] * x[["nsx(size, 3)1:grade2"]] +
    betas[["nsx(size, 3)2:grade2"]] * x[["nsx(size, 3)2:grade2"]] +
    betas[["nsx(size, 3)3:grade2"]] * x[["nsx(size, 3)3:grade2"]] +
    betas[["nsx(size, 3)1:grade3"]] * x[["nsx(size, 3)1:grade3"]] +
    betas[["nsx(size, 3)2:grade3"]] * x[["nsx(size, 3)2:grade3"]] +
    betas[["nsx(size, 3)3:grade3"]] * x[["nsx(size, 3)3:grade3"]] +
    betas[["nsx(estrg_recp, 3)1:log(eventtime)"]] * x[["nsx(estrg_recp, 3)1"]] * log(t) +
    betas[["nsx(estrg_recp, 3)2:log(eventtime)"]] * x[["nsx(estrg_recp, 3)2"]] * log(t) +
    betas[["nsx(estrg_recp, 3)3:log(eventtime)"]] * x[["nsx(estrg_recp, 3)3"]] * log(t)
  
  # Return the log cumulative hazard at time t
  res
}

# ----------------------------------------------------------------
## Function for data generation according to the true model
# ----------------------------------------------------------------

gen_data <- function(true_mod) {
  cov <- as.data.frame(true_mod@x)
  # Simulate the event times
  print("simulate data ... ")
  dat <- simsurv(betas = coef(true_mod),      # "true" parameter values
                 x = cov,                     # covariate data 
                 logcumhazard = logcumhaz,    # definition of log cum hazard
                 maxt = NULL,                 # no administrative right-censoring
                 interval = c(1E-8,10000000)) # interval for root finding
  table(dat$status)
  print("data simulated")
  
  expon <- flexsurvreg(Surv(eventtime, status) ~ 1, data=dat, dist="exponential")
  expon
  
  Tcen <- -log(runif(nrow(dat)))/(3*exp(expon$coefficients)) # 75% censoring
  
  Tobs <- ifelse(dat$eventtime > Tcen, Tcen, dat$eventtime)
  dat$status <- ifelse(dat$eventtime > Tcen, 0, 1)
  table(dat$status)/length(dat$status)
  
  # Merge the simulated event times onto covariate data frame
  dat <- cbind(gbcs.exp[,5:12], dat)
  dat <- as.data.frame(dat)
  dat$train <- gbcs.exp$train
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
  
  ## data generation
  dat <- gen_data(true_mod)
  
  ## training data
  dataleft.tot <-dat[dat$train,]
  dim(dataleft.tot)
  ## sample training data
  sample.train <- sample.int(nrow(dataleft.tot), size = 1000)    
  dataleft <- dataleft.tot[sample.train,]
  
  table(dataleft$status)/nrow(dataleft)
  
  ## validatation/test data
  dataright <- dat[!dat$train,]
  
  ## different models for predictions
  print("Cox and AFT models ")
  
  mod.cox <- coxph(Surv(eventtime, status) ~ age + menopause + hormone + size + grade + nodes + 
                     prog_recp + estrg_recp, data = dataleft, x=TRUE)
  
  mod.cox.nl <- try(coxph(Surv(eventtime, status) ~ ns(age,3) + menopause + hormone + ns(size,3) + grade + nodes + 
                          ns(prog_recp, 3) + ns(estrg_recp,3), data = dataleft, x=TRUE))

  mod.cox.nl.int <- try(coxph(Surv(eventtime, status) ~ ns(age,3) + menopause + hormone + ns(size,3) * grade + nodes + 
                              ns(prog_recp, 3) + ns(estrg_recp,3), data = dataleft, x=TRUE))

  # true model
  mod.stpm <- try(stpm2(Surv(eventtime, status)~nsx(age,3) + menopause + hormone + nsx(size,3) * grade + nodes + 
                             nsx(prog_recp,3) + nsx(estrg_recp, 3) , data=dataleft,
                             smooth.formula=~nsx(log(eventtime), df=3) + nsx(estrg_recp, 3) : log(eventtime)))

  print("Cox and flexible models estimated ... ")

  ## Boosting
  print("Boosting Cox and AFT models")
  mod.mb <- gamboost(Surv(eventtime, status) ~ bbs(age) + bols(menopause) + bols(hormone) +
                          bbs(size) + bols(grade) +
                          bbs(nodes) + bbs(prog_recp) + bbs(estrg_recp), data = dataleft, family=CoxPH(),
                          control = boost_control(mstop = 1000, center = TRUE))

  cv10f <- cv(model.weights(mod.mb), type = "kfold")
  cvm <- cvrisk(mod.mb, folds = cv10f, papply = lapply, grid=seq(100, 1000, 100))
  mod.mb[mstop(cvm)]

  ## Random Forest SRC
  
  print("Random Forest SRC")
  
  TUNING <- randomForestSRC::tune(Surv(eventtime, status) ~ age + menopause + hormone + size + grade +
                                    nodes + prog_recp + estrg_recp, data = dataleft, ntreeTry = 100)
  
  mod.rf <- rfsrc.fast(Surv(eventtime, status) ~ age + menopause + hormone + size + grade +
                       nodes + prog_recp + estrg_recp, data = dataleft, mtry=TUNING$optimal[2],
                       nodesize = TUNING$optimal[1], ntree = 5000, ntime=250, forest=TRUE)
  
  ## data preparation (scaling and one-hot encoding) for Neural Network
  ## Training data
  
  print("PC hazard DNN")
  
  dataleft2 <- dataleft
  dataleft2$age <- scale(dataleft2$age)
  dataleft2$menopause <- dataleft2$menopause-1
  dataleft2$hormone <- dataleft2$hormone-1
  dataleft2$size <- scale(dataleft2$size)
  dataleft2$grade <- as.factor(dataleft2$grade)
  temp <- one_hot(as.data.table(dataleft2$grade) )
  dataleft2$G1 <- temp$V1_1
  dataleft2$G2 <- temp$V1_2
  dataleft2$G3 <- temp$V1_3
  dataleft2$nodes <- scale(dataleft2$nodes)
  dataleft2$prog_recp <- scale(dataleft2$prog_recp)
  dataleft2$estrg_recp <- scale(dataleft2$estrg_recp)
  dataleft3 <- dataleft2[, -c(5, 9, 12)]
  head(dataleft3)
  
  CENTER=c(
    attr(dataleft2$age, "scaled:center"),
    attr(dataleft2$size, "scaled:center"),
    attr(dataleft2$nodes, "scaled:center"),
    attr(dataleft2$prog_recp, "scaled:center"),
    attr(dataleft2$estrg_recp, "scaled:center"))
  SCALE=c(
    attr(dataleft2$age, "scaled:scale"),
    attr(dataleft2$size, "scaled:scale"),
    attr(dataleft2$nodes, "scaled:scale"),
    attr(dataleft2$prog_recp, "scaled:scale"),
    attr(dataleft2$estrg_recp, "scaled:scale"))
  
  newdata2 <- dataright
  newdata2$age <- scale(newdata2$age, center=CENTER[1], scale = SCALE[1])
  newdata2$menopause <- newdata2$menopause-1
  newdata2$hormone <- newdata2$hormone-1
  newdata2$size <- scale(newdata2$size, center=CENTER[2], scale = SCALE[2])
  newdata2$grade <- as.factor(newdata2$grade)
  temp <- one_hot(as.data.table(newdata2$grade) )
  newdata2$G1 <- as.numeric(temp$V1_1)
  newdata2$G2 <- as.numeric(temp$V1_2)
  newdata2$G3 <- as.numeric(temp$V1_3)
  newdata2$nodes <- scale(newdata2$nodes, center=CENTER[3], scale = SCALE[3])
  newdata2$prog_recp <- scale(newdata2$prog_recp, center=CENTER[4], scale = SCALE[4])
  newdata2$estrg_recp <- scale(newdata2$estrg_recp, center=CENTER[5], scale = SCALE[5])
  dataright3 <- as.data.frame(newdata2[, -c(5, 9, 12)])
  
  task_gbcs = TaskSurv$new(id = "dataleft3", backend = dataleft3,
                           time = "eventtime", event = "status")
  
  search_space <- ps(
    ## p_dbl for numeric valued parameters
    dropout = p_dbl(lower = 0.01, upper = .1),
    weight_decay = p_dbl(lower = 0.0001, upper = 0.001),
    learning_rate = p_dbl(lower = 0.01, upper = .1),
    ## p_int for integer valued parameters
    nodes = p_int(lower = 64, upper = 128),
    k = p_int(lower = 1, upper = 4)
  )
  
  search_space$trafo <- function(x, param_set) {
    x$num_nodes = rep(x$nodes, x$k)
    x$nodes = x$k = NULL
    return(x)
  }
  
  ## load learner
  learner.dh <- lrn("surv.pchazard",
                    frac = 0.3, early_stopping = TRUE, epochs = 10000,
                    optimizer = "adam" 
  )
  
  at <- AutoTuner$new(
    learner = learner.dh,
    search_space = search_space,
    resampling = rsmp("cv", folds = 3),
    measure = msr("surv.cindex"), # msr("surv.graf", times=2190, integrated=FALSE)
    terminator = trm("evals", n_evals = 10),
    tuner = tnr("random_search")
  )
  
  at$train(task_gbcs)
  
  
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
  predictRisk.args = list(Learner = list(CENTER[1], CENTER[2], CENTER[3], CENTER[4], CENTER[5],
                                         SCALE[1],  SCALE[2], SCALE[3], SCALE[4], SCALE[5]))
  )
  
  return(list(res = res))
  }

res <- sim_run(true_mod)

res$res$Brier$score



