# Load necessary libraries ------------------------------------------------
library(readr)
library(ggplot2)
library(dplyr)
library(neuralnet)
library(h2o)
h2o.init(nthreads = -1)

# Simulate variables -----------------------------------------------------
#Simulate variables
#set.seed(1) #For replicability
grid <- expand.grid(X=seq(0,1,by=0.05), Y=seq(0,1,by=0.05))
grid2 <- mutate(grid,
                Z0f=(Y<X))
grid2$Z0n <- as.integer(grid2$Z0)

#Plot  
ggplot(grid2,aes(X,Y,color=Z0f)) + geom_point()

# Example of linearly separable data -----------------------------------------------------
#ANN Example of decision boundary
#Note that by design the variables are already scaled
n <- names(grid2)
f <- as.formula(paste("Z0f ~", paste(n[c(1,2)], collapse = " + ")))
nn <- neuralnet(f,data=grid2,hidden=1,linear.output=T)

#Predict
nn.results <- compute(nn,grid2[,1:2])

#Round results and add to dataframe grid2 for plotting
grid2$Z0hat <- round(nn.results$net.result)
#Plot decision boundary
ggplot(grid2,aes(X,Y,z=Z0hat)) + geom_contour(binwidth = 0.8) + geom_point(aes(X,Y,color = Z0f))

# Example of non-linearly separable data -----------------------------------------------------
#Non-linearly separable data
grid3 <- mutate(grid,
                Z0f=(((0.2*Y-0.4)^2 + (X-0.5)^2) > 0.15))
grid3$Z0n <- as.integer(grid2$Z0f)

#Plot  
ggplot(grid3,aes(X,Y,color=Z0f)) + geom_point()

#ANN Example of decision boundary
#Note that by design the variables are already scaled
nn <- neuralnet(f,data=grid3,hidden=1,linear.output=T)

#Predict
nn.results <- compute(nn,grid3[,1:2])

#Round results and add to dataframe grid3 for plotting
grid3$Z0hat <- round(nn.results$net.result)
#Plot decision boundary
ggplot(grid3,aes(X,Y,z=Z0hat)) + geom_contour(binwidth = 0.8) + geom_point(aes(X,Y,color = Z0f))

#Increase number of hidden neurones
nn <- neuralnet(f,data=grid3,hidden=2,linear.output=T)

#Predict
nn.results <- compute(nn,grid3[,1:2])

#Round results and add to dataframe grid3 for plotting
grid3$Z0hat <- round(nn.results$net.result)
#Plot decision boundary
ggplot(grid3,aes(X,Y,z=Z0hat)) + geom_contour(binwidth = 0.5) + geom_point(aes(X,Y,color = Z0f))

#Increase number of layers
nn <- neuralnet(f,data=grid3,hidden=c(1,1),linear.output=T)

#Predict
nn.results <- compute(nn,grid3[,1:2])

#Round results and add to dataframe grid3 for plotting
grid3$Z0hat <- round(nn.results$net.result)
#Plot decision boundary
ggplot(grid3,aes(X,Y,z=Z0hat)) + geom_contour(binwidth = 0.5) + geom_point(aes(X,Y,color = Z0f))

#Increase number of layers and neurones
nn <- neuralnet(f,data=grid3,hidden=c(2,1),linear.output=T)

#Predict
nn.results <- compute(nn,grid3[,1:2])

#Round results and add to dataframe grid3 for plotting
grid3$Z0hat <- round(nn.results$net.result)
#Plot decision boundary
ggplot(grid3,aes(X,Y,z=Z0hat)) + geom_contour(binwidth = 0.5) + geom_point(aes(X,Y,color = Z0f))

# Example of non-linearly separable data -----------------------------------------------------
#Non-linearly separable data
grid3 <- mutate(grid,
                Z0f=as.factor((((Y-1)^2 + (X-1)^2) > 0.20) + (((Y)^2 + (X)^2) > 0.20) + (((Y-1)^2 + (X)^2) > 0.20)))
grid3$Z0n <- as.integer(grid3$Z0f)
grid3$Z0f <- grid3$Z0f==2

#Plot  
ggplot(grid3,aes(X,Y,color=Z0f)) + geom_point()

#ANN Example of decision boundary
#Note that by design the variables are already scaled
nn <- neuralnet(f,data=grid3,hidden=1,linear.output=T)

#Predict
nn.results <- compute(nn,grid3[,1:2])

#Round results and add to dataframe grid3 for plotting
grid3$Z0hat <- round(nn.results$net.result)
#Plot decision boundary
ggplot(grid3,aes(X,Y,z=Z0hat)) + geom_contour(binwidth = 0.8) + geom_point(aes(X,Y,color = Z0f))

#Increase number of hidden neurones
nn <- neuralnet(f,data=grid3,hidden=2,linear.output=T)

#Predict
nn.results <- compute(nn,grid3[,1:2])

#Round results and add to dataframe grid3 for plotting
grid3$Z0hat <- round(nn.results$net.result)
#Plot decision boundary
ggplot(grid3,aes(X,Y,z=Z0hat)) + geom_contour(binwidth = 0.5) + geom_point(aes(X,Y,color = Z0f))

#Increase number of layers
nn <- neuralnet(f,data=grid3,hidden=c(1,1),linear.output=T)

#Predict
nn.results <- compute(nn,grid3[,1:2])

#Round results and add to dataframe grid3 for plotting
grid3$Z0hat <- round(nn.results$net.result)
#Plot decision boundary
ggplot(grid3,aes(X,Y,z=Z0hat)) + geom_contour(binwidth = 0.5) + geom_point(aes(X,Y,color = Z0f))

#Increase number of layers and neurones
nn <- neuralnet(f,data=grid3,hidden=c(1,2),linear.output=T)

#Predict
nn.results <- compute(nn,grid3[,1:2])

#Round results and add to dataframe grid3 for plotting
grid3$Z0hat <- round(nn.results$net.result)
#Plot decision boundary
ggplot(grid3,aes(X,Y,z=Z0hat)) + geom_contour(binwidth = 0.5) + geom_point(aes(X,Y,color = Z0f))

#Increase number of layers and neurones
nn <- neuralnet(f,data=grid3,hidden=c(2,2),linear.output=T)

#Predict
nn.results <- compute(nn,grid3[,1:2])

#Round results and add to dataframe grid3 for plotting
grid3$Z0hat <- round(nn.results$net.result)
#Plot decision boundary
ggplot(grid3,aes(X,Y,z=Z0hat)) + geom_contour(binwidth = 0.5) + geom_point(aes(X,Y,color = Z0f))

#Increase number of layers and neurones
nn <- neuralnet(f,data=grid3,hidden=c(4,6,6,6,4),linear.output=T)

#Predict
nn.results <- compute(nn,grid3[,1:2])

#Round results and add to dataframe grid3 for plotting
grid3$Z0hat <- round(nn.results$net.result)
#Plot decision boundary
ggplot(grid3,aes(X,Y,z=Z0hat)) + geom_contour(binwidth = 0.5) + geom_point(aes(X,Y,color = Z0f))


#Another example
n <- names(dataset)
f <- as.formula(paste("Exited ~", paste(c("CreditScore","Age"), collapse = " + ")))

df.train <- data.frame(training_set$Exited,training_set$CreditScore,training_set$Age)
names(df.train)[1] <- "Exited"
names(df.train)[2] <- "CreditScore"
names(df.train)[3] <- "Age"

df.test <- data.frame(test_set$Exited,test_set$CreditScore,test_set$Age)
names(df.test)[1] <- "Exited"
names(df.test)[2] <- "CreditScore"
names(df.test)[3] <- "Age"

nn <- neuralnet(f,data=df.train,hidden=c(2,1),linear.output=T)

#Predict
nn.results <- compute(nn,df.test[,2:3])

#Round results and add to dataframe grid3 for plotting
df.test$Exitedhat <- round(nn.results$net.result)
#Plot decision boundary
ggplot(df.test,aes(CreditScore,Age,z=Exitedhat)) + geom_contour(binwidth = 0.5) + geom_point(aes(CreditScore,Age,color = as.factor(Exited)))
