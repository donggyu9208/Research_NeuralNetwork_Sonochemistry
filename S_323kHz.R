## Loads neuralnet package
library('neuralnet')
library('caret')

# Data
temperature <- c(5,	5,	5,	5,	5,	20,	20,	20,	20,	20,	35,	35,	35,	35,	35, 50,	50,	50,	50,	50,	10,	10,	10,	10,	10)
power <- c(13,	22,	33,	43,	55,	13,	22,	33,	43,	55,	13,	22,	33,	43,	55,	13,	22,	33,	43,	55,	13,	22,	33,	43,	55)
gastype <- data.frame(var = c("helium", "air", "O2", "Ar", "Ar/O2", "air", "O2", "Ar", "Ar/O2", "helium", "O2", "Ar", "Ar/O2", "helium", "air", "Ar", "Ar/O2", "helium", "air", "O2", "Ar/O2", "helium", "air", "O2", "Ar"))
gas <- createDummyFeatures(gastype, cols = "var")
names(gas)[1]<- paste("air")
names(gas)[2]<- paste("Ar")
names(gas)[3]<- paste("Ar/O2")
names(gas)[4]<- paste("He")
names(gas)[5]<- paste("O2")
air <-(gas[,1])
Ar <- (gas[,2])
Ar_O2 <- (gas[,3])
He <- (gas[,4])
O2 <- (gas[,5])

frequency_323 <-rep(323, 25) # kHz
surfate_conc_323 <- c(0.1846,	0.1131,	0.1288,	0.0571,	0.0105,	0.0538,	0.0565,	0.1646,	0.0135,	0.0807,	0.1238,	0.0481,	0.0878,	0.0564,	0.0525,	0.1247,	0.0891,	0.1257,	0.0523,	0.0565,	0.0854,	0.0521,	0.1157,	0.0505,	0.0986)


# Put them in a table
full_data <- data.frame(frequency = frequency_323, temperatrue = temperature, power = power, air = air, Ar = Ar, Ar_O2 = Ar_O2, He = He, O2 = O2, surfate_conc = surfate_conc_323)
net.data <- neuralnet(surfate_conc ~ frequency + temperature + power + air + Ar + Ar_O2 + He + O2, full_data, hidden = 1, err.fct = 'sse', act.fct =
                        'logistic', linear.output = FALSE, learningrate = 0.001)
min_error <- net.data$result.matrix[1,]

# algorithm for finding the number of neurons that give the best result
for (neuron in 2:100) {
    net.data <- neuralnet(surfate_conc ~ frequency + temperature + power + air + Ar + Ar_O2 + He + O2, 
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

net.data <- neuralnet(surfate_conc ~ frequency + temperature + power + air + Ar + Ar_O2 + He + O2, full_data, hidden = best_neuron, err.fct = 'sse', act.fct =
                                    'logistic', linear.output = FALSE, learningrate = 0.001)
best_neuron

min_error <- net.data$result.matrix[1,]
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
