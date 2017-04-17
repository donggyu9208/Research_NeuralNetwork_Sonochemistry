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
  
  frequency_323 <-rep(323, 25) # [=] kHz
  power_323 <- c(13,	22,	33,	43,	55,	13,	22,	33,	43,	55,	13,	22,	33,	43,	55,	13,	22,	33,	43,	55,	13,	22,	33,	43,	55) # [=] W
  surfate_conc_323 <- c(0.1846,	0.1131,	0.1288,	0.0571,	0.0105,	0.0538,	0.0565,	0.1646,	0.0135,	0.0807,	0.1238,	0.0481,	0.0878,	0.0564,	0.0525,	0.1247,	0.0891,	0.1257,	0.0523,	0.0565,	0.0854,	0.0521,	0.1157,	0.0505,	0.0986) # [=] mM
  # Put all reqindividual data in a table
  full_data <- data.frame(frequency = frequency_323, temperature = temperature, power = power_323, air = air, Ar = Ar, Ar_O2 = Ar_O2, He = He, O2 = O2, surfate_conc = surfate_conc_323)
  
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
  scaled_df <- data.frame(df[,1]* 10,df[,2]* 10) # Scaling back to the original 
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
  temp_323_meanS <- data.frame(c(5, 10, 20, 35, 50), c(temp_5, temp_10, temp_20, temp_35, temp_50))
  names(temp_323_meanS)[1]<- paste("Temperature (C)")
  names(temp_323_meanS)[2]<- paste("mean Sulfate (mM)")
  
  ## Power
  power_13 <- mean(c(scaled_df[1, 1], scaled_df[6, 1], scaled_df[11, 1], scaled_df[16, 1], scaled_df[21, 1]))
  power_22 <- mean(c(scaled_df[2, 1], scaled_df[7, 1], scaled_df[12, 1], scaled_df[17, 1], scaled_df[22, 1]))
  power_33 <- mean(c(scaled_df[3, 1], scaled_df[8, 1], scaled_df[13, 1], scaled_df[18, 1], scaled_df[23, 1]))
  power_43 <- mean(c(scaled_df[4, 1], scaled_df[9, 1], scaled_df[14, 1], scaled_df[19, 1], scaled_df[24, 1]))
  power_55 <- mean(c(scaled_df[5, 1], scaled_df[10, 1], scaled_df[15, 1], scaled_df[20, 1], scaled_df[25, 1]))
  power_323_meanS <- data.frame(c(13, 22, 33, 43, 55), c(power_13, power_22, power_33, power_43, power_55))
  names(power_323_meanS)[1]<- paste("Power (W)")
  names(power_323_meanS)[2]<- paste("mean Sulfate (mM)")
  
  ## Gas
  gas_He <- mean(c(scaled_df[1, 1], scaled_df[10, 1], scaled_df[14, 1], scaled_df[18, 1], scaled_df[22, 1]))
  gas_Air <- mean(c(scaled_df[2, 1], scaled_df[6, 1], scaled_df[15, 1], scaled_df[19, 1], scaled_df[23, 1]))
  gas_O2 <- mean(c(scaled_df[3, 1], scaled_df[7, 1], scaled_df[11, 1], scaled_df[20, 1], scaled_df[24, 1]))
  gas_Ar <- mean(c(scaled_df[4, 1], scaled_df[8, 1], scaled_df[12, 1], scaled_df[16, 1], scaled_df[25, 1]))
  gas_Ar_O2 <- mean(c(scaled_df[5, 1], scaled_df[9, 1], scaled_df[13, 1], scaled_df[17, 1], scaled_df[21, 1]))
  gas_323_meanS <- data.frame(c('He', 'Air', 'O2', 'Ar', 'Ar/O2'), c(gas_He, gas_Air, gas_O2, gas_Ar, gas_Ar_O2))
  names(gas_323_meanS)[1]<- paste("Gas")
  names(gas_323_meanS)[2]<- paste("mean Sulfate (mM)")
  
  min_error
  best_neuron
  best_learning_rate
  temp_323_meanS 
  power_323_meanS
  gas_323_meanS
  
  #################################### Experimental Data #####################################################
  temp_323_exp <- data.frame(c(5, 10, 20, 35, 50), c(0.9882, 0.8046, 0.7382, 0.7372, 0.8966))
  power_323_exp <- data.frame(c(13, 22, 33, 43, 55), c(1.1446, 0.7178, 1.2452, 0.4596, 0.5976))
  gas_323_exp <- data.frame(c('He', 'Air', 'O2', 'Ar', 'Ar/O2'), c(0.9990, 0.7748, 0.8322, 0.9862, 0.5726))
  
  temp_323_both <- data.frame(temp = temp_323_exp[, 1], predicted = temp_323_meanS[, 2], experimental = temp_323_exp[, 2])
  power_323_both <- data.frame(power = power_323_exp[, 1], predicted = power_323_meanS[, 2], experimental = power_323_exp[, 2])
  gas_323_both <-data.frame(gas = gas_323_exp[, 1], predicted = gas_323_meanS[, 2], experimental = gas_323_exp[, 2])
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
  names(temp_323_both) <- c("Temperature", "Predicted", "Experimental")
  temp_323_both.melt <- melt(temp_323_both, id.vars = 'Temperature')
  temp_323_graph <- (
  ggplot(temp_323_both.melt, aes(x = Temperature, 
                                 y = value, 
                                 shape = variable, 
                                 color = variable))
  + geom_point() 
  + geom_xspline(spline_shape=-0.4,size= 0.7)
  + scale_x_continuous(name = "Temperature (C)", breaks = 0:6 * 10, limits = c(0, 55))
  + scale_y_continuous(name = "mean sulfate (mM)", breaks = 0:24 * 0.05, limits = c(0.7, 1))
  #+ scale_color_manual(values = c("Predicted" = 'red','Experimental' = 'blue')) 
  + scale_shape_manual(values = c('Predicted' = 17, 'Experimental' = 16))
  + labs(color = "", shape = "")
  )
  
  ## Power
  names(power_323_both) <- c("Power", "Predicted", "Experimental")
  power_323_both.melt <- melt(power_323_both, id.vars = 'Power')
  power_323_graph <- (
    ggplot(power_323_both.melt, aes(x = Power, 
                                   y = value, 
                                   shape = variable, 
                                   color = variable))
    + geom_point() 
    + geom_xspline(spline_shape=-0.4,size= 0.7)
    + scale_x_continuous(name = "Power (W)", breaks = 0:6 * 10, limits = c(10, 55))
    + scale_y_continuous(name = "mean sulfate (mM)", breaks = 0:7 * 0.2, limits = c(0.4, 1.3))
    #+ scale_color_manual(values = c("Predicted" = 'red','Experimental' = 'blue')) 
    + scale_shape_manual(values = c('Predicted' = 17, 'Experimental' = 16))
    + labs(color = "", shape = "")
  )
  
  ## Gas
  gas_323_both.melt <- melt(gas_323_both, id.vars = 'gas')
  gas_323_graph <- (
  ggplot(data=gas_323_both.melt, aes(x=gas, y=value, fill=variable)) 
    + geom_bar(stat="identity", position=position_dodge())
    + scale_y_continuous(name = "mean sulfate (mM)", breaks = 0:7 * 0.2, limits = c(0, 1.1))
    + labs(fill = "")
    + coord_cartesian(ylim=c(0, 1.05))
  )
  