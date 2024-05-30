# load packages 
library(lme4); library(lmerTest); library(emmeans); library(pbkrtest)

# load behavioural data
data_performance <- read.csv("data/behaviour/behaviour_performance.csv")
data_rtime <- read.csv("data/behaviour/behaviour_reactiontime.csv")

# fit lmmodel - performance
lmmmod <- lmer(dependentVar ~ adapter*contrast + (1|subject), data=data_performance, REML = TRUE)
anova(lmmmod)
emmeans(lmmmod, list(pairwise ~ adapter|contrast))

# fit lmmodel - reaction time
lmmmod <- lmer(dependentVar ~ adapter*contrast + (1|subject), data=data_rtime, REML = TRUE)
anova(lmmmod)
emmeans(lmmmod, list(pairwise ~ adapter|contrast))
