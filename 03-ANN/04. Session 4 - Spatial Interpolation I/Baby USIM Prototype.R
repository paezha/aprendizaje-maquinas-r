# Baby USIM Prototype

# Clear environment
rm(list = ls())

#Load libraries
library(spdep)
library(adespatial)
library(dplyr)
library(ggplot2)
library(keras)
#library(rgdal)
#library(rgeos)
library(tmap)

# Create a grid
nc <- 20/2 # Number of columns
nr <- 40/2 # Number of rows
no <- nc * nr # Number of observations
coords <- expand.grid(x=seq(1, 2 * nc,by=2), y=seq(1, 2 * nr,by=2))
ID <- seq(1, nr * nc, by=1)

# find the k = 8 nearest neighbors
g8nn <- knn2nb(knearneigh(as.matrix(coords), k = 8), row.names = ID)
# create a listw object (an adjacency matrix in list form)
g8listw <- nb2listw(g8nn)

# Calculate eigenvectors
EV <- mem(g8listw)
# Retrieve number of maps
nm <- ncol(EV)
# Normalize the eigenvectors
maxs <- apply(EV, 2, max)
mins <- apply(EV, 2, min)
EV <- as.data.frame(scale(EV, center = mins, scale = maxs - mins))
# Append the coordinates
EV <- cbind(coords, EV)
# Plot eigenvector
VEC <- 3
ggplot(data = EV,
       aes(x = x,
           y = y)) +
  geom_tile(aes(fill = EV[,VEC+2])) +
  coord_fixed(ratio = 1) +
  scale_fill_distiller(palette = "Oranges",
                       direction = 1)

# Split in train and test sets
set.seed(10)
list <- sort(sample(1:ncol(EV)-2, (nr * nc)/3))

# Calculate Moran's coefficient for eigenvectors
mc <- numeric()
for(i in 3:ncol(EV)){
  junk <- moran(EV[,i], g8listw, length(g8nn), Szero(g8listw))
  mc[i-2] <- as.numeric(junk[1])
}
# Plot Moran's coefficients
plot(mc)
# Split in train and test sets
mc.train <- mc[-list]
mc.test <- mc[list]

# Subset train and test samples
coords.train <- coords[-list,]
coords.test <- coords[list,]
# Number of training observations
no.train <- no - nrow(coords.test)
# Number of testing observations
no.test <- no - nrow(coords.train)
# Select a vector to plot
VEC <- 1
data <- EV[-list, 1:2]
data$VAR <- EV[-list, VEC + 2]
ggplot(data = data, aes(x = x, y = y)) + geom_tile(aes(fill = VAR)) + coord_fixed(ratio = 1) + scale_fill_distiller(palette = "Oranges", direction = -1)

# Organize training data by nearest neighbors
# Find the k = 8 nearest neighbors
g8nn.train <- knn2nb(knearneigh(as.matrix(coords.train), k = 8), row.names = ID[-list])
# Find the distance to the k = 8 nearest neighbors
d8nn.train <- nbdists(g8nn.train,as.matrix(coords.train))
# Select the maps for training
VARS <- EV[,3:ncol(EV)]
# Use all maps
#VARS <- VARS[,-list]
# Number of maps used for training
nm.training <- ncol(VARS)
# These arrays will contain nm maps for training, each with no.train observations
VAR8nn.train <- array(data = NA, dim = c(nm.training * no.train,8))
VAR.train <- array(data = NA, dim = c(nm.training * no.train))
MC.train <- array(data = NA, dim = c(nm.training * no.train))
count <- 0
for(j in 1:nm.training){
  # Select map and observations for creating the training data set
  evec <- VARS[,j]
  evec <- evec[-list]
  for(i in 1:no.train){
    count <- count + 1
    VAR.train[count] <- evec[i] # This is the outcome variable, the value to predict
    # This is the input MC for training
    MC.train[count] <- mc.train[j]
    # These are the input observations for training
    VAR8nn.train[count,] <- VARS[attributes(g8nn.train)$region.id[g8nn.train[[i]]], j]
  }
  print(j)
}

ggplot(data = EV[attributes(g8nn.train)$region.id[g8nn.train[[1]]],], aes(x = x, y = y)) + geom_tile(aes(fill = MEM1)) + coord_fixed(ratio = 1) + scale_fill_distiller(palette = "Oranges", direction = -1)

# Build the network

#Main input layer: 8 nearest neighbors of training data
main_input <- layer_input(shape = c(8), name = "main_input")

# # First dense network
# dense1_output <- main_input |>
#   layer_dense(units = 8, activation = "softmax", input_shape = c(8)) |>
#   layer_dense(units = 30, activation = 'relu') |>
#   layer_dense(units = 30, activation = 'softmax') |>
#   layer_dense(units = 20, activation = 'relu') |>
#   layer_dense(units = 20, activation = 'softmax')

# First dense network
dense1_output <- main_input |>
  layer_dense(units = 8, activation = "softmax", input_shape = c(8)) |>
  layer_dense(units = 30, activation = 'relu') |>
  layer_dense(units = 30, activation = 'relu') |>
  layer_dense(units = 20, activation = 'relu') |>
  layer_dense(units = 20, activation = 'relu') |>
  layer_dense(units = 15, activation = 'relu') |>
  layer_dense(units = 15, activation = 'relu') |>
  layer_dense(units = 10, activation = 'relu') |>
  layer_dense(units = 10, activation = 'relu')

# Auxiliary output
auxiliary_output <- dense1_output |>
  layer_dense(units = 1, activation = "relu", name = "aux_output")

# # Auxiliary input layer: Moran's coefficient of input map
# auxiliary_input <- layer_input(shape = c(1), name = "aux_input")
#
# # Main output: concatenate the auxiliary input to the output of dense1 output
# dense2_output <- layer_concatenate(c(dense1_output, auxiliary_input)) |>
#   layer_dense(units = 64, activation = "softmax") |>
#   layer_dense(units = 64, activation = "relu") |>
#   layer_dense(units = 64, activation = "softmax") |>
#   layer_dense(units = 1, activation = "relu", name = "main_output")

# usim <- keras_model(
#   inputs = c(main_input,auxiliary_input),
#   outputs = c(dense2_output, auxiliary_output)
# )

usim <- keras_model(
  inputs = main_input,
  outputs = auxiliary_output
)

summary(usim)

# Compile model
# usim |> compile(
#   optimizer = "adagrad",
#   loss = "mae",
#   loss_weights = c(1.0, 0.2),
#   metrics = c("mae", "acc")
# )

usim |> compile(
  optimizer = "adagrad",
  loss = "mae",
  metrics = c("mae", "acc")
)

# Train model
# Train map to use for predictions
# mapnum <- 1
# bottom <- (mapnum - 1) * 200 + 1
# top <- mapnum * 200
# usim |> fit(
#   x = list(VAR8nn.train[bottom:top,], MC.train[bottom:top]),
#   y = list(VAR.train[bottom:top], VAR.train[bottom:top]),
#   epochs = 20,
#   batch_size = 32
# )

#Train map to use for predictions
mapnum <- 3
bottom <- (mapnum - 1) * no.train + 1
top <- mapnum * no.train
usim |> fit(
  x = VAR8nn.train[bottom:top,],
  y = VAR.train[bottom:top],
  epochs = 300,
  batch_size = 32
)

# ## Test the model
# # Assemble list of nearest neighbors and observations for test observations
# # Select the maps for testing
# VARS <- EV[,3:ncol(EV)]
# VARS <- VARS[,list]
# nm.testing <- ncol(VARS)
# VAR8nn.test <- array(data = NA, dim = c(nm.testing * no,8))
# VAR.test <- array(data = NA, dim = c(nm.testing * no))
# MC.test <- array(data = NA, dim = c(nm.testing * no))
# count <- 0
# for(j in 1:nm.testing){
#   # Select map and observations for creating the training data set
#   evec <- VARS[,j]
#   for(i in 1:no){
#     count <- count + 1
#     VAR.test[count] <- evec[i]
#     MC.test[count] <- mc.train[j]
#     VAR8nn.test[count,] <- VARS[attributes(g8nn)$region.id[g8nn[[i]]], j]
#   }
#   print(j)
# }

# # Evauate the model
# score <- usim |> evaluate(x = list(VAR8nn.train, MC.train),
#                            y = list(VAR.train, VAR.train),
#                            batch_size = 10)
# score

# Predict
# Select test map to use for predictions
# mapnum <- 66
# bottom <- (mapnum - 1) * 200 + 1
# top <- mapnum * 200
# #pred_sp <- usim |> predict(x = list(VAR8nn.test[bottom:top,], MC.test[bottom:top]), batch_size = 10)
# pred_sp <- usim |> predict(x = VAR8nn.test[bottom:top,], batch_size = 10)
# #Plot predictions against observations
# comp_sp <- as.data.frame(cbind(VAR.test[bottom:top],pred_sp))
# comp_sp <- rename(comp_sp, VAR.test = V1, predicted = V2)
# ggplot(data = comp_sp, aes(x = VAR.test, y = predicted)) + geom_point() + geom_abline(intercept = 0, slope = 1)
# cor(comp_sp)

# Predict
# Organize test data by nearest neighbors
# Find the k = 8 nearest neighbors for test observations
id.junk <- ID[-list]
coords.junk <- coords.train # Coordinates of train observations
g8nn.test <- list() # Initialize list of nearest neighbors
for(i in 1:no.test){
  id.junk[no.train + 1] <- ID[list[i]] # Retrieve the region.id for the ith item in the test sample
  coords.junk[no.train + 1,] <- coords[list[i],] # Append the coordinates of the ith item of test sample
  junk <- knn2nb(knearneigh(as.matrix(coords.junk), k = 8), row.names = id.junk) # Find nearest neighbors
  #NOTE: This list of nearest neighbors is built using the region.ids
  g8nn.test[[i]] <- attributes(junk)$region.id[junk[[no.train + 1]]] # Retrieve nearest neighbors of ith item in test sample
  print(i)
}
# Assemble list of nearest neighbors and observations for test observations
VAR <- EV[,mapnum + 2] #EV$MEM2
VAR.test <- VAR[list]
VAR8nn.test <- array(data = NA, dim = c(no.test,8))
for(i in 1:no.test){
  VAR8nn.test[i,] <- VAR[g8nn.test[[i]]]
}
data = EV[,1:2]
data$VAR = EV[,mapnum + 2]
ggplot(data = data[g8nn.test[[1]],], aes(x = x, y = y)) + geom_tile(aes(fill = VAR)) + coord_fixed(ratio = 1) + scale_fill_distiller(palette = "Oranges", direction = -1)

pred_sp <- usim |> predict(VAR8nn.test, batch_size = 10)
#Plot predictions against observations
comp_sp <- as.data.frame(cbind(VAR.test,pred_sp))
comp_sp <- rename(comp_sp, predicted = V2)
ggplot(data = comp_sp, aes(x = VAR.test, y = predicted)) + geom_point() + geom_abline(intercept = 0, slope = 1)
cor(VAR.test,pred_sp)


##
# Out of sample experiment: use a completely different zoning system
taz <- st_read("Hamilton CMA tts06.shp")
taz.centroids <- gCentroid(taz, byid = TRUE)
taz.centroids <- SpatialPointsDataFrame(taz.centroids, taz@data)

# Organize training data by nearest neighbors
# Find the k = 8 nearest neighbors
taz8nn <- knn2nb(knearneigh(taz.centroids, k = 8, longlat = TRUE), row.names = taz.centroids$ID)
# create a listw object (an adjacency matrix in list form)
taz8listw <- nb2listw(taz8nn)

# Calculate eigenvectors
tazEV <- mem(taz8listw)
# Retrieve number of maps
taznm <- ncol(tazEV)
# Normalize the eigenvectors
maxs <- apply(tazEV, 2, max)
mins <- apply(tazEV, 2, min)
tazEV <- as.data.frame(scale(tazEV, center = mins, scale = maxs - mins))
# Append eigenvectors to SpatialPolygonsDataFrame
taz@data <- cbind(taz@data, tazEV)

# Plot eigenvector
VEC <- paste0("MEM", 180)
tm_shape(taz) + tm_polygons(VEC, style = "quantile") + tm_compass() + tm_scale_bar()

#Retrieve map for testing
mapnum <- 1
tazVAR8nn <- array(data = NA, dim = c(297,8))
for(i in 1:nrow(tazEV)){
  tazVAR8nn[i,] <- tazEV[taz8nn[[i]], mapnum]
}

#Prediction
mapnum <- 4
pred_sp <- usim |> predict(x = tazVAR8nn, batch_size = 10)
#Plot predictions against observations
comp_sp <- as.data.frame(cbind(tazEV[,mapnum],pred_sp))
comp_sp <- rename(comp_sp, VAR.test = V1, predicted = V2)
ggplot(data = comp_sp, aes(x = VAR.test, y = predicted)) + geom_point() + geom_abline(intercept = 0, slope = 1)
cor(comp_sp)





## TEST
# Organize training data by nearest neighbors
# Find the k = 8 nearest neighbors
g8nn.train <- knn2nb(knearneigh(as.matrix(coords.train), k = 8), row.names = ID[-list])
# Find the distance to the k = 8 nearest neighbors
d8nn.train <- nbdists(g8nn.train,as.matrix(coords.train))
# Select the maps for training
VARS <- EV[,3:ncol(EV)]
# Use all maps
#VARS <- VARS[,-list]
# Number of maps used for training
nm.training <- ncol(VARS)
# These arrays will contain nm maps for training, each with no.train observations
#VAR8nn.train <- array(data = NA, dim = c(nm.training * no.train,8,2))
VAR8nn.train <- array(data = NA, dim = c(nm.training * no.train,8))
VAR.train <- array(data = NA, dim = c(nm.training * no.train))
MC.train <- array(data = NA, dim = c(nm.training * no.train))
count <- 0
for(j in 1:nm.training){
  # Select map and observations for creating the training data set
  evec <- VARS[,j]
  evec <- evec[-list]
  for(i in 1:no.train){
    count <- count + 1
    VAR.train[count] <- evec[i]
    MC.train[count] <- mc.train[j]
    VAR8nn.train[count,] <- VARS[attributes(g8nn.train)$region.id[g8nn.train[[i]]], j]
    #VAR8nn.train[count,,1] <- VARS[attributes(g8nn.train)$region.id[g8nn.train[[i]]], j]
    #VAR8nn.train[count,,2] <- unlist(d8nn.train[i])
  }
  print(j)
}

# Build a network
# Create model
model_sp <- keras_model_sequential()

# Define and compile the model
model_sp |>
  layer_dense(units = 8, activation = "softmax", input_shape = c(8)) |>
  layer_dense(units = 30, activation = 'softmax') |>
  layer_dense(units = 30, activation = 'softmax') |>
  layer_dense(units = 20, activation = 'softmax') |>
  layer_dense(units = 20, activation = 'relu') |>
  layer_dense(units = 15, activation = 'relu') |>
  layer_dense(units = 15, activation = 'relu') |>
  layer_dense(units = 10, activation = 'relu') |>
  layer_dense(units = 10, activation = 'relu') |>
  layer_dense(units = 1, activation = 'relu')

# Summary of model
summary(model_sp)

model_sp |>
  compile(
    optimizer = 'adam',
    loss = 'mse',
    metrics = "acc")

# save_model_hdf5(model_sp, "model_sp_8s_20r_20s_15r_15s_1r", overwrite = TRUE,
#                 include_optimizer = TRUE)

# Train the model
history <- model_sp |> fit(VAR8nn.train, VAR.train, batch_size = 10, epochs = 20)

model_sp |>
  compile(
    optimizer = 'adam',
    loss = 'mse',
    metrics = "acc")

# save_model_hdf5(model_sp, "model_sp_8s_20r_20s_15r_15s_1r", overwrite = TRUE,
#                 include_optimizer = TRUE)

# Train the model
history <- model_sp |> fit(VAR8nn.train, VAR.train, batch_size = 10, epochs = 20)
