set.seed(500)
library(MASS)
data <- Boston

index <- sample(1:nrow(data),round(0.75*nrow(data)))
train <- data[index,]
test <- data[-index,]
lm.fit <- glm(medv~., data=train)
summary(lm.fit)
pr.lm <- predict(lm.fit,test)
MSE.lm <- sum((pr.lm - test$medv)^2)/nrow(test)

maxs <- apply(data, 2, max) 
mins <- apply(data, 2, min)

scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))

train_ <- scaled[index,]
test_ <- scaled[-index,]

library(neuralnet)
n <- names(train_)
f <- as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))
nn <- neuralnet(f,data=train_,hidden=c(5,3),linear.output=T)


# Example with only one datum and a simple network with only one hidden neuron
datum <- filter(data,(medv==8.5)&(age==85.4))
datum <- as.data.frame(scale(datum, center = mins, scale = maxs - mins))
nn.datum <- neuralnet(f,data=datum,hidden=1,linear.output=T)
plot(nn.datum)

covariates <- c(1,nn.datum$covariate)
weights1 <- unlist(nn.datum$weights[[1]][1])
weights2 <- unlist(nn.datum$weights[[1]][2])

#Note that by default the activation function is the logistic. Therefore, the weighted sum needs to be applied
sumWxX1 <- sum(weights1 * covariates)
v1 <- 1/(1 + exp(-sumWxX1))

#Note that the activation function of the output layer is the identity
sumWxX2 <- weights2[1] + weights2[2] * v1

# Example with only one datum and a simple network with two hidden neurons
nn.datum2 <- neuralnet(f,data=datum,hidden=2,linear.output=T)
plot(nn.datum2)
