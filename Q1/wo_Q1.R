## inload the library 
library(cluster)
library(dendextend)
library(factoextra)
library(caret)
library(FactoMineR)
library(corrplot)
library(sp)
library(gstat)
library(plyr)
library(magrittr)
library(ggplot2)
library(automap)
library(e1071)
library(lattice)
library(caret)
library(plyr)
library(foreach)
library(Cubist)
library(corrplot)
library(PerformanceAnalytics)
library(raster)
library(sp)
library(rgdal)

## subset the time for different time period
AllDayGeo<-function(data){
  data7<-data.frame(time=data$V1,Lon=data$V15,Lat=data$V16)
  data8<-data.frame(time=data$V1,Lon=data$V17,Lat=data$V18)
  data9<-data.frame(time=data$V1,Lon=data$V19,Lat=data$V20)
  data10<-data.frame(time=data$V1,Lon=data$V21,Lat=data$V22)
  data11<-data.frame(time=data$V1,Lon=data$V23,Lat=data$V24)
  data12<-data.frame(time=data$V1,Lon=data$V25,Lat=data$V26)
  
  data18<-data.frame(time=data$V1,Lon=data$V37,Lat=data$V38)
  data19<-data.frame(time=data$V1,Lon=data$V39,Lat=data$V40)
  data20<-data.frame(time=data$V1,Lon=data$V41,Lat=data$V42)
  data21<-data.frame(time=data$V1,Lon=data$V43,Lat=data$V44)
  data22<-data.frame(time=data$V1,Lon=data$V45,Lat=data$V46)
  data23<-data.frame(time=data$V1,Lon=data$V47,Lat=data$V48)
  data24<-data.frame(time=data$V1,Lon=data$V49,Lat=data$V50)
  
  data13<-data.frame(time=data$V1,Lon=data$V27,Lat=data$V28)
  data14<-data.frame(time=data$V1,Lon=data$V29,Lat=data$V30)
  data15<-data.frame(time=data$V1,Lon=data$V31,Lat=data$V32)
  data16<-data.frame(time=data$V1,Lon=data$V33,Lat=data$V34)
  data17<-data.frame(time=data$V1,Lon=data$V35,Lat=data$V36)
  X1<-rbind(data7,data8,data9,data10,data11,data12,data13,data14,data15,data16,data17,data18,data19,data20,data21,data22,data23,data24)
  X1<-na.omit(X1)
  X1<-X1[order(X1$time),]
  return(X1)
}


FiveColokGeo<-function(data){
  data5<-data.frame(time=data$V1,Lon=data$V35,Lat=data$V36)
  X1<-rbind(data5)
  X1<-na.omit(X1)
  X1<-X1[order(X1$time),]
  return(X1)
}


SixColokGeo<-function(data){
  data6<-data.frame(time=data$V1,Lon=data$V37,Lat=data$V38)
  X1<-rbind(data6)
  X1<-na.omit(X1)
  X1<-X1[order(X1$time),]
  return(X1)
}

SevenColockGeo<-function(data){
  data7<-data.frame(time=data$V1,Lon=data$V39,Lat=data$V40)
  X1<-rbind(data7)
  X1<-na.omit(X1)
  X1<-X1[order(X1$time),]
  return(X1)
}


EightclockGeo<-function(data){
  data8<-data.frame(time=data$V1,Lon=data$V41,Lat=data$V42)
  X1<-rbind(data8)
  X1<-na.omit(X1)
  X1<-X1[order(X1$time),]
  return(X1)
}


AfternoonGeo<-function(data){
  data13<-data.frame(time=data$V1,Lon=data$V27,Lat=data$V28)
  data14<-data.frame(time=data$V1,Lon=data$V29,Lat=data$V30)
  data15<-data.frame(time=data$V1,Lon=data$V31,Lat=data$V32)
  data16<-data.frame(time=data$V1,Lon=data$V33,Lat=data$V34)
  data17<-data.frame(time=data$V1,Lon=data$V35,Lat=data$V36)
  X1<-rbind(data13,data14,data15,data16,data17)
  X1<-na.omit(X1)
  X1<-X1[order(X1$time),]
  return(X1)
  }

MorningGeo<-function(data){
  data7<-data.frame(time=data$V1,Lon=data$V15,Lat=data$V16)
  data8<-data.frame(time=data$V1,Lon=data$V17,Lat=data$V18)
  data9<-data.frame(time=data$V1,Lon=data$V19,Lat=data$V20)
  data10<-data.frame(time=data$V1,Lon=data$V21,Lat=data$V22)
  data11<-data.frame(time=data$V1,Lon=data$V23,Lat=data$V24)
  data12<-data.frame(time=data$V1,Lon=data$V25,Lat=data$V26)
  x1<-rbind(data7,data8,data9,data10,data11,data12)
  X1<-na.omit(X1)
  X1<-X1[order(X1$time),]
  return(X1)
}

## transfer LongLat to UTM
LongLatToUTM<-function(data){
  xy <- data.frame(time =data$time, X = data$Lon, Y = data$Lat)
  coordinates(xy) <- c("X", "Y")
  proj4string(xy) <- CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
  res <- spTransform(xy, CRS(paste("+proj=utm +zone=",51," ellps=WGS84",sep='')))
  return(as.data.frame(res))
}

## read the data 
data<-read.csv("E://Benya//modelling0504//wo2.csv",header=T)
Geoinf<-read.csv("E://Benya//modelling0504//shujudasai_1.csv",header = F)

## scale the data 
IMEI<-data$IMEI
data2<-as.data.frame(scale(data[,-1], center = TRUE, scale = TRUE))

wo_plus2 <- cbind(IMEI,data2)

## find data hot points
HighFreq<-lapply(names(wo_plus2)[-1], function(i) {wo_plus2[wo_plus2[i]>1000,]})

BankMessages<-subset(Geoinf,V2 %in% HighFreq[[1]][,1])
AutoApp<-subset(Geoinf,V2 %in% HighFreq[[2]][,1])
FinanApp<-subset(Geoinf,V2 %in% HighFreq[[3]][,1])
StockApp<-subset(Geoinf,V2 %in% HighFreq[[4]][,1])
SocialNetworkSize<-subset(Geoinf,V2 %in% HighFreq[[5]][,1])
CrossProvinceExp<-subset(Geoinf,V2 %in% HighFreq[[6]][,1])
OverseaExp<-subset(Geoinf,V2 %in% HighFreq[[7]][,1])
browseShoppingSites<-subset(Geoinf,V2 %in% HighFreq[[8]][,1])
browseITSites<-subset(Geoinf,V2 %in% HighFreq[[9]][,1])
browseRestaurantSites<-subset(Geoinf,V2 %in% HighFreq[[10]][,1])
browseRealEstateSites<-subset(Geoinf,V2 %in% HighFreq[[11]][,1])
browseHealthSites<-subset(Geoinf,V2 %in% HighFreq[[12]][,1])
browseFinanceSites<-subset(Geoinf,V2 %in% HighFreq[[13]][,1])
browseTravelSites<-subset(Geoinf,V2 %in% HighFreq[[14]][,1])
browseSportsSites<-subset(Geoinf,V2 %in% HighFreq[[15]][,1])
browseAutoSites<-subset(Geoinf,V2 %in% HighFreq[[16]][,1])
browseNewsSites<-subset(Geoinf,V2 %in% HighFreq[[17]][,1])
browseCommunitySites<-subset(Geoinf,V2 %in% HighFreq[[18]][,1])
browseRecreationSites<-subset(Geoinf,V2 %in% HighFreq[[19]][,1])
browseJobsSites<-subset(Geoinf,V2 %in% HighFreq[[20]][,1])
browseEducationSites<-subset(Geoinf,V2 %in% HighFreq[[21]][,1])
browseOnlineGamingSites<-subset(Geoinf,V2 %in% HighFreq[[22]][,1])

unique(AutoApp[order(AutoApp$V2),][,2])

## plot the correlation plot
cor.mat <- round(cor(Rich),2)
corrplot(cor.mat, type="upper", order="hclust", tl.col="black", tl.srt=45,na.rm=T)

BankMessages2<-BankMessages[order(BankMessages$V1),]


## select rich peolpe

Rich<-wo_plus2[(wo_plus2$OverseaExp>1),]

cor.mat <- round(cor(Rich[,-c(1,8,23)]),2)
corrplot(cor.mat, type="upper", order="hclust", tl.col="black", tl.srt=45,na.rm=T)


## seprate the data into different group 
Auto<-Rich[((Rich$AutoApp >1)| (Rich$browseAutoSites >1)),]
Auto$auto<-Auto$AutoApp+Auto$browseAutoSites

Shopping<-Rich[(Rich$browseShoppingSites>1),]

Education<-Rich[((Rich$browseJobsSites>1) | (Rich$browseEducationSites >1)),]
Education$education<-Education$browseJobsSites+Education$browseEducationSites

Lifestyle<-Rich[((Rich$browseHealthSites >1)| (Rich$browseSportsSites>1) |(Rich$browseCommunitySites>1)),]
Lifestyle$lifestyle<-Lifestyle$browseHealthSites+Lifestyle$browseSportsSites+Lifestyle$browseCommunitySites

Recreation<-Rich[((Rich$browseRestaurantSites>1) | (Rich$browseTravelSites >1)|(Rich$browseRecreationSites>1)),]
Recreation$recreation<-Recreation$browseRestaurantSites+Recreation$browseTravelSites+Recreation$browseRecreationSites

Fin<-Rich[((Rich$BankMessages>5) |(Rich$FinanApp >5) | ( Rich$StockApp>5) |(Rich$browseRealEstateSites>5) | (Rich$browseFinanceSites>5)),]
Fin$Fin<-Fin$BankMessages+Fin$FinanApp+Fin$StockApp+Fin$browseRealEstateSites+Fin$browseFinanceSites

## set up geoinformation for five group 
ShoppingGeo<-subset(Geoinf,V2 %in% Shopping$IMEI)
EducationGeo<-subset(Geoinf,V2 %in% Education$IMEI)
LifestyleGeo<-subset(Geoinf,V2 %in% Lifestyle$IMEI)
RecreationGeo<-subset(Geoinf,V2 %in% Recreation$IMEI)
FinGeo<-subset(Geoinf,V2 %in% Fin$IMEI)
AutoGeo<-subset(Geoinf,V2 %in% Auto$IMEI)

SunLifeEven<-LifestyleLocation[((LifestyleLocation$time=="20151227")|(LifestyleLocation$time=="20160103")),]

## set up time period and transfer the reproject system for the data 
## shopping 

## morning 


## evening

shoppingEven<-EveningGeo(ShoppingGeo)

## export the data to ArcGis and create the map

## education 

EducaLoa<-AllDayGeo(EducationGeo)

## lifestle

LifestyleLoa<-AllDayGeo(LifestyleGeo)

## recreation
recreationLOC<-AllDayGeo(RecreationGeo)

## Auto 
AutoLOC<-AllDayGeo(AutoGeo)

FINLOC<-AllDayGeo(FinGeo)





## random sample 1% from the dataset
set.seed(111)
bound <- floor((nrow(wo_plus2)/100)*1) 
# sample row 
wo_plus3 <- wo_plus2[sample(nrow(wo_plus2)), ]  
wo_plus.train <- wo_plus3[1:bound, ]


##Hierarchical Clustering
distance <- dist(wo_plus.train, method = "euclidean")
res.hc <- hclust(distance, method = "ward.D2" )

plot_hc<-function(data,k){
  grp <- cutree(res.hc, k = k)
  fviz_cluster(list(data = data, cluster = grp,na.rm=T))+theme_bw()
}

plot_hc(wo_plus.train,7)


## PCA
res.pca <- PCA(wo_plus2, ncp = 4, graph = FALSE)
## plot PCA results
fviz_eig(res.pca)
fviz_pca_var(res.pca)




worker<-Geoinf[Geoinf$V31>121.490,]



worker<-subset(Geoinf , V31 > 121.493 & V31 < 121.509 & v32>31.231 & v32<31.244)


a <- rle(sort(as.vector(worker$V2)))
worker.num<-data.frame(a$lengths,a$values)

worker.num<-worker.num[worker.num$a.lengths>1,]


target<-read.csv("E://Benya//modelling0504//GIS//PuDongWorker.csv",header = T)
worker<-subset(Geoinf,V2 %in% target$a.values)




