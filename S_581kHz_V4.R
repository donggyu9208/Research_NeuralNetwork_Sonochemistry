## Loads neuralnet package
library('dplyr')
library('neuralnet')
library('mlr')
library('caret')
library('ggplot2')
library('ggalt')
library('reshape2')


######################################### RAW DATA ########################################

temperature <- c(5,	5,	5,	5,	5,	20,	20,	20,	20,	20,	35,	35,	35,	35,	35, 50,	50,	50,	50,	50,	10,	10,	10,	10,	10) # [=] C
gastype <- data.frame(var = c("helium", "air", "O2", "Ar", "Ar/O2", "air", "O2", "Ar", "Ar/O2", "helium", "O2", "Ar", "Ar/O2", "helium", "air", "Ar", "Ar/O2", "helium", "air", "O2", "Ar/O2", "helium", "air", "O2", "Ar"))
gas <- createDummyFeatures(gastype, cols = "var") # Creates a dummy variable for categorical types

# Individualize each gas types
air <-(gas[,1])
Ar <- (gas[,2])
Ar_O2 <- (gas[,3])
He <- (gas[,4])
O2 <- (gas[,5])

frequency_581 <-rep(581, 25) # [=] kHz
power_581 <- c(49,	81,	117,	153,	185,	49,	81,	117,	153,	185,	49,	81,	117,	153,	185,	49,	81,	117,	153,	185,	49,	81,	117,	153,	185)
surfate_conc_581 <- c(0.059,	0.0548,	0.1327,	0.064,	0.0193,	0.0341,	0.1289,	0.0484,	0.0022,	0.1621,	0.1028,	0.1158,	0.0695,	0.0355	,0.1012	,0.1597,	0.0161	,0.1028	,0.0354,	0.06,	0.0925,	0.0519	,0.0583	,0.1166,	0.0584)
# Put all reqindividual data in a table
full_data <- data.frame(frequency = frequency_581, temperature = temperature, power = power_581, air = air, Ar = Ar, Ar_O2 = Ar_O2, He = He, O2 = O2, surfate_conc = surfate_conc_581)

############################### RUNNING NEURAL NETWORK ###########################################

min_error <- 1
learning_rate <- c(0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0001)

for (lr_i in 1:length(learning_rate)) {
  curr_learningRate = learning_rate[lr_i]
  for (neuron in 1:100) {
    net.data <- neuralnet(surfate_conc ~ frequency + temperature + power + air + Ar + Ar_O2 + He + O2,
                          full_data, hidden = neuron, err.fct = 'sse', act.fct = 'logistic', linear.output = FALSE,
                          learningrate = curr_learningRate)
    error <- net.data$result.matrix[1,]
    if (min_error > error) {
      min_error <- error
      best_neuron <- neuron
      best_learning_rate <- curr_learningRate
      best_startweights <- net.data$startweights
    }
  }
}


############################# RESULTING DATA ##################################

net.data <- neuralnet(surfate_conc ~ frequency + temperature + power + air + Ar + Ar_O2 + He + O2, full_data, 
                      hidden = best_neuron, startweights = best_startweights, 
                      err.fct = 'sse', act.fct = 'logistic', linear.output = FALSE, learningrate = best_learning_rate)

net.data$net.result
net.data$data$surfate_conc
net.data$result.matrix[1,]
predict.data <- neuralnet::compute(net.data, full_data[,1:8])
predict.data$net.result
df <- data.frame(net.data$net.result, net.data$response)
scaled_df <- data.frame(df[,1] * 10,df[,2] * 10) # Scaling back to the original 
names(scaled_df)[1]<- paste("predicted")
names(scaled_df)[2]<- paste("desired")
scaled_df


########################################################## Calculating mean_sulfate###########################
## Temperature
temp_5 <- mean(scaled_df[1:5, 1])
temp_20 <- mean(scaled_df[6:10, 1])
temp_35 <- mean(scaled_df[11:15, 1])
temp_50 <- mean(scaled_df[16:20, 1])
temp_10 <- mean(scaled_df[21:25, 1])
temp_581_meanS <- data.frame(c(5, 10, 20, 35, 50), c(temp_5, temp_10, temp_20, temp_35, temp_50))
names(temp_581_meanS)[1]<- paste("Temperature (C)")
names(temp_581_meanS)[2]<- paste("mean Sulfate (mM)")

## Power
power_49 <- mean(c(scaled_df[1, 1], scaled_df[6, 1], scaled_df[11, 1], scaled_df[16, 1], scaled_df[21, 1]))
power_81 <- mean(c(scaled_df[2, 1], scaled_df[7, 1], scaled_df[12, 1], scaled_df[17, 1], scaled_df[22, 1]))
power_117 <- mean(c(scaled_df[3, 1], scaled_df[8, 1], scaled_df[13, 1], scaled_df[18, 1], scaled_df[23, 1]))
power_153 <- mean(c(scaled_df[4, 1], scaled_df[9, 1], scaled_df[14, 1], scaled_df[19, 1], scaled_df[24, 1]))
power_185 <- mean(c(scaled_df[5, 1], scaled_df[10, 1], scaled_df[15, 1], scaled_df[20, 1], scaled_df[25, 1]))
power_581_meanS <- data.frame(c(49, 81, 117, 153, 185), c(power_49, power_81, power_117, power_153, power_185))
names(power_581_meanS)[1]<- paste("Power (W)")
names(power_581_meanS)[2]<- paste("mean Sulfate (mM)")

## Gas
gas_He <- mean(c(scaled_df[1, 1], scaled_df[10, 1], scaled_df[14, 1], scaled_df[18, 1], scaled_df[22, 1]))
gas_Air <- mean(c(scaled_df[2, 1], scaled_df[6, 1], scaled_df[15, 1], scaled_df[19, 1], scaled_df[23, 1]))
gas_O2 <- mean(c(scaled_df[3, 1], scaled_df[7, 1], scaled_df[11, 1], scaled_df[20, 1], scaled_df[24, 1]))
gas_Ar <- mean(c(scaled_df[4, 1], scaled_df[8, 1], scaled_df[12, 1], scaled_df[16, 1], scaled_df[25, 1]))
gas_Ar_O2 <- mean(c(scaled_df[5, 1], scaled_df[9, 1], scaled_df[13, 1], scaled_df[17, 1], scaled_df[21, 1]))
gas_581_meanS <- data.frame(c('He', 'Air', 'O2', 'Ar', 'Ar/O2'), c(gas_He, gas_Air, gas_O2, gas_Ar, gas_Ar_O2))
names(gas_581_meanS)[1]<- paste("Gas")
names(gas_581_meanS)[2]<- paste("mean Sulfate (mM)")

min_error
best_neuron
best_learning_rate
temp_581_meanS 
power_581_meanS
gas_581_meanS

#################################### Experimental Data #####################################################
temp_581_exp <- data.frame(c(5, 10, 20, 35, 50), c(0.6596, 0.7554, 0.7514, 0.8496, 0.7480))
power_581_exp <- data.frame(c(49, 81, 117, 153, 185), c(0.8962, 0.7350, 0.8234, 0.5074, 0.8020))
gas_581_exp <- data.frame(c('He', 'Air', 'O2', 'Ar', 'Ar/O2'), c(0.8226, 0.5676, 1.0820, 0.8926, 0.3992))

temp_581_both <- data.frame(temp = temp_581_exp[, 1], predicted = temp_581_meanS[, 2], experimental = temp_581_exp[, 2])
power_581_both <- data.frame(power = power_581_exp[, 1], predicted = power_581_meanS[, 2], experimental = power_581_exp[, 2])
gas_581_both <-data.frame(gas = gas_581_exp[, 1], predicted = gas_581_meanS[, 2], experimental = gas_581_exp[, 2])
#####################################MULTIPLE GRAPH FUNCTION ###############################
# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

################################### Graph Predictive vs Actual ###########################
## Width: 781
## Height: 384
##
## Temperature
names(temp_581_both) <- c("Temperature", "Predicted", "Experimental")
temp_581_both.melt <- melt(temp_581_both, id.vars = 'Temperature')
temp_581_graph <- (
  ggplot(temp_581_both.melt, aes(x = Temperature, 
                                 y = value, 
                                 shape = variable, 
                                 color = variable))
  + geom_point() 
  + geom_xspline(spline_shape=-0.4,size= 0.7)
  + scale_x_continuous(name = "Temperature (C)", breaks = 0:6 * 10, limits = c(0, 55))
  + scale_y_continuous(name = "mean sulfate (mM)", breaks = 0:24 * 0.05, limits = c(0.65, 0.875))
  #+ scale_color_manual(values = c("Predicted" = 'red','Experimental' = 'blue')) 
  + scale_shape_manual(values = c('Predicted' = 17, 'Experimental' = 16))
  + labs(color = "", shape = "")
)

## Power
names(power_581_both) <- c("Power", "Predicted", "Experimental")
power_581_both.melt <- melt(power_581_both, id.vars = 'Power')
power_581_graph <- (
  ggplot(power_581_both.melt, aes(x = Power, 
                                  y = value, 
                                  shape = variable, 
                                  color = variable))
  + geom_point(size = 2) 
  + geom_xspline(spline_shape=-0.4,size= 0.7)
  + scale_x_continuous(name = "Power (W)", breaks = 0:6 * 50, limits = c(20, 200))
  + scale_y_continuous(name = "mean sulfate (mM)", breaks = 0:10 * 0.1, limits = c(0.5, 0.9))
  #+ scale_color_manual(values = c("Predicted" = 'red','Experimental' = 'blue')) 
  + scale_shape_manual(values = c('Predicted' = 17, 'Experimental' = 16))
  + labs(color = "", shape = "")
)

## Gas
gas_581_both.melt <- melt(gas_581_both, id.vars = 'gas')
power_581_graph <- (
  ggplot(data=gas_581_both.melt, aes(x=gas, y=value, fill=variable)) 
  + geom_bar(stat="identity", position=position_dodge())
  + scale_y_continuous(name = "mean sulfate (mM)", breaks = 0:7 * 0.2, limits = c(0, 1.1))
  + labs(fill = "")
  + coord_cartesian(ylim=c(0, 1.1))
)
