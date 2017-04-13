## Loads neuralnet package
library('dplyr')
library('neuralnet')
library('mlr')
library('caret')

############################### RAW DATA ########################################

temperature <- c(5,	5,	5,	5,	5,	20,	20,	20,	20,	20,	35,	35,	35,	35,	35, 50,	50,	50,	50,	50,	10,	10,	10,	10,	10) # [=] C
gastype <- data.frame(var = c("helium", "air", "O2", "Ar", "Ar/O2", "air", "O2", "Ar", "Ar/O2", "helium", "O2", "Ar", "Ar/O2", "helium", "air", "Ar", "Ar/O2", "helium", "air", "O2", "Ar/O2", "helium", "air", "O2", "Ar"))
gas <- createDummyFeatures(gastype, cols = "var") # Creates a dummy variable for categorical types
# names(gas)[1]<- paste("air") # Names each column of gas table
# names(gas)[2]<- paste("Ar")
# names(gas)[3]<- paste("Ar/O2")
# names(gas)[4]<- paste("He")
# names(gas)[5]<- paste("O2")

# Individualize each gas types
air <-(gas[,1])
Ar <- (gas[,2])
Ar_O2 <- (gas[,3])
He <- (gas[,4])
O2 <- (gas[,5])

frequency_323 <-rep(323, 25) # [=] kHz
power_323 <- c(13,	22,	33,	43,	55,	13,	22,	33,	43,	55,	13,	22,	33,	43,	55,	13,	22,	33,	43,	55,	13,	22,	33,	43,	55) # [=] W
surfate_conc_323 <- c(0.1846,	0.1131,	0.1288,	0.0571,	0.0105,	0.0538,	0.0565,	0.1646,	0.0135,	0.0807,	0.1238,	0.0481,	0.0878,	0.0564,	0.0525,	0.1247,	0.0891,	0.1257,	0.0523,	0.0565,	0.0854,	0.0521,	0.1157,	0.0505,	0.0986) # [=] mM
# Put all reqindividual data in a table
full_data <- data.frame(frequency = frequency_323, temperature = temperature, power = power_323, air = air, Ar = Ar, Ar_O2 = Ar_O2, He = He, O2 = O2, surfate_conc = surfate_conc_323)

############################### RUNNING NEURAL NETWORK ###########################################
# 80 neurons in a hidden layer, learning rate = 0.05, activation function = logistic function, error function = squared sum error
net.data <- neuralnet(surfate_conc ~ frequency + temperature + power + air + Ar + Ar_O2 + He + O2, full_data, 
                      hidden = 80, err.fct = 'sse', act.fct = 'logistic', linear.output = FALSE, learningrate = 0.05)
min_error <- net.data$result.matrix[1,]

############################### K-FOLD CROSS VALIDATION ######################################

k <- 10
## 90% of the sample size
training_sample_size <- floor(0.9 * nrow(full_data)) ## 90% of training set

## set the seed to make your partition reproductible
set.seed(1)
## This randomly shuffles the data
cv_error_array <- NULL
shuffled_data <- full_data[sample(nrow(full_data)),]

#Create k-equally sized folds
folds <- cut(seq(1,nrow(shuffled_data)),breaks=k,labels=FALSE)

for(i in 1:k){
#Segement your data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  test_data <- full_data[testIndexes, ]
  train_data <- full_data[-testIndexes, ]
  net_data <- neuralnet(surfate_conc ~ frequency + temperature + power + air + Ar + Ar_O2 + He + O2, train_Data, 
                        hidden = 80, err.fct = 'sse', act.fct = 'logistic', linear.output = FALSE, learningrate = 0.05)
  predict.data <- neuralnet::compute(net.data, test_data[,1:8]) ## Predicts the surfate_conc with the test_data parameters
  predict_result <- predict.data$net.result*(max(full_data$surfate_conc) - min(full_data$surfate_conc)) + min(max(full_data$surfate_conc))
  test_result <- (test_data$surfate_conc)*(max(full_data$surfate_conc)-min(full_data$surfate_conc))+min(full_data$surfate_conc)
  cv_error_array[i] <- sum((test_result - predict_result)^2)/nrow(test_data)
}

# Calculate mean error for cross fold validation

cv_error <- mean(cv_error_array)


# algorithm for finding the number of neurons that give the best result
for (neuron in 2:100) {
    net_data <- neuralnet(surfate_conc ~ frequency + temperature + power + air + Ar + Ar_O2 + He + O2, 
                          full_data, hidden = neuron, err.fct = 'sse', act.fct =
                          'logistic', linear.output = FALSE, learningrate = 0.001)
    error <- net.data$result.matrix[1,]
    if (min_error > error) {
        min_error <- error
        best_neuron <- neuron
    }
}

best_neuron #62 71 99 92
min_error#

net_data <- neuralnet(surfate_conc ~ frequency + temperature + power + air + Ar + Ar_O2 + He + O2, full_data, hidden = best_neuron, err.fct = 'sse', act.fct =
                                    'logistic', linear.output = FALSE, learningrate = 0.05)
best_neuron

min_error <- net_data$result.matrix[1,]
min_error
df <- data.frame(net.data$net.result, net.data$response)
corrected_df <- data.frame(df[,1]/100,df[,2]/100) # Scaling back to the original 
names(corrected_df)[1]<- paste("predicted")
names(corrected_df)[2]<- paste("desired")

cat("The optimal number of neurons for 323kHz is: ", best_neuron)
cat("Minimum error with sum squared error method is: ", min_error)
print("The final result table is as follows: ")

corrected_df

plot(net.data)
