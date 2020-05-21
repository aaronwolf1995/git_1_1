library(tidyverse)
library(stringi)
library(stringr)
#储存变量
load("20200515.RData")
# 分别读取三个数据集
dia1 <- read.csv("diabetes.csv",header = TRUE,sep = ",", stringsAsFactors = F)
dia1 <- as.data.frame(dia1)
dia1$datasource <- paste0("kaggle",1:nrow(dia1))
table(is.na(dia1))
View(dia1)

dia2 <- read.csv("total1_diabetes_zhangyao_1.csv",header = TRUE,sep = ",", stringsAsFactors = F)
dia2 <- as.data.frame(dia2)
dia2$datasource <- paste0("paper",1:nrow(dia2))
colnames(dia2)[1] <- "sex"
table(is.na(dia2))
dia2 <- dia2[,-grep(pattern="change",x=colnames(dia2))]
dim(dia2)
View(dia2)

dia3 <- read.csv("total2_diabetes_zhangyao_2.csv",header = TRUE,sep = ",", stringsAsFactors = F)
dia3 <- as.data.frame(dia3)
dia3$datasource <- paste0("paper",(1:nrow(dia3))+3981)
colnames(dia3)[1] <- colnames(dia2)[3]
colnames(dia3)[2] <- colnames(dia2)[1]
dia3 <- dia3[,-grep(pattern="change",x=colnames(dia3))]
table(is.na(dia3))
View(dia3)


#挑选其中一部分变量,去除包含change字符串的
DIA1 <- dia1 %>% 
  select(datasource,Age,BMI,BloodPressure,
         SkinThickness,Insulin,DiabetesPedigreeFunction,Outcome)
head(DIA1)

DIA2 <- dia2 %>% 
  select(datasource,sex,age1,bmi_1sd:Diabete_family_history1,Diabetes_incidence)
head(DIA2)
str(DIA2)
DIA4 <- DIA2[,as.factor(DIA2[,c(2,3,9:12,15:16)])]

DIA2 %>% 
  group_by(Diabetes_incidence) %>% 
  summarize(count=n())
save(dia1,dia2,dia3,DIA1,DIA2,DIA3,file = "20200515.RData")


DIA2_2 <- DIA2[DIA2$Diabetes_incidence==0,] 
DIA2_2 <- DIA2_2[sample(1:nrow(DIA2_2),sum(DIA2$Diabetes_incidence==1)*2),]
nrow(DIA2_2)
rownames(DIA2_2)=1:nrow(DIA2_2)

# 将DIA2中Diabetes_incidence中为1的和“0的部分”挑选出来
DIA2_3 <- DIA2 %>% 
  group_by(Diabetes_incidence) %>% 
  arrange(desc(Diabetes_incidence)) 

DIA2_4 <- DIA2_3[1:(sum(DIA2_3$Diabetes_incidence==1)*3),]
nrow(DIA2_4)
View(DIA2_4)

write.csv(DIA2_4,file = "DIA2_4.csv")

# 20200515
# 想办法合并数据框,为什么rbind函数不行？
mmm <- merge(DIA2_1,DIA2_2,by="datasource")
identical(colnames(DIA2_1),colnames(DIA2_2))
View(DIA2_2)
View(mmm)















