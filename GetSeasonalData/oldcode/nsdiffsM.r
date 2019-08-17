require(forecast)
rawdat <- read.csv('in.csv',header=FALSE)
X <- as.numeric(rawdat)
for (i in 1:8){
    cat(i)
    print(paste("The year is", i))
    data <-nsdiffs(ts(X,frequency = 61))
}
write.csv(data,'out.csv')