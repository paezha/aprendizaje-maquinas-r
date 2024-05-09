#Example Walker Lake, uses 'Walker Lake.csv' data

#Load libraries
library(readr)
library(spdep)
library(ggplot2)

#Read csv file
walker <- read_csv("walker Lake.csv")
summary(walker)

#Create list of neighbors and create weights matrix
listW6 <- nb2listw(knn2nb(knearneigh(data.matrix(walker[,2:3]), k=6)))
W6 <- nb2mat(knn2nb(knearneigh(data.matrix(walker[,2:3]), k=6)),style="B")

#Dependent variable is centered on the mean
y<-walker$V-mean(walker$V)

#Initialize starting values of parameters
teta <- c(0.3,0.1,sd(y),0.4)

#Source the functions
source("ofns.R")
source("trans.R")
source("clasifica.R")

#Estimate the model, retrieve parameters
res.pars <- optim(teta,fn=ofns,gr=NULL,y=y,W=listW6,standard=NULL)
pars<- trans(res.pars$par)
pars
#Calculate probabilities of autoregressive parameter for two states
probs <- clasifica(res.pars$par,y,W=listW6,standard=NULL)

#Plot probabilities
walker["Probs"] <- probs
p <- ggplot(walker,aes(X,Y)) + geom_point(aes(color=Probs,size=Probs))
p
