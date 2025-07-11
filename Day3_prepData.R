library(dataReporter)

#Load data (download from https://www.projectdatasphere.org/projectdatasphere/html/pcdc):
  data <- read.csv("./data/Coretable_training.csv", header = TRUE,
                    na.strings = c(".", "", "MISSING", "Missing"))

#Data cleaning
  #Recode binary "YES"/NA variables to 1/0 (NA means NO = 0):

  #Missing means "NO" for a number of variables:
   for (i in 55:131) {
     levels(data[,i]) <- 1
     data[, i] <- as.numeric(data[, i])
     data[is.na(data[,i]),i] <- 0
   }

  #DEATH: Replace NA by "CENSORED"
  levels(data$DEATH) <- c("YES", "CENSORED")
  data$DEATH[is.na(data$DEATH)] <- "CENSORED"   
      
  #Race: Define new variable with two categories (white/non-white)
  #1 is non-white
  data$RACE_nonwhite <- NA
  data$RACE_nonwhite[data$RACE_C == "White"] <- 0
  data$RACE_nonwhite[data$RACE_C != "White"] <- 1
  
  
  #AGE: Define new dummy-coded age category variables
  #Reference group is 18-64
  data$AGEGRP_65to74 <- NA
  data$AGEGRP_65to74[data$AGEGRP2 == "65-74"] <- 1
  data$AGEGRP_65to74[data$AGEGRP2 != "65-74"] <- 0
  data$AGEGRP_75plus <- NA
  data$AGEGRP_75plus[data$AGEGRP2 == ">=75"] <- 1
  data$AGEGRP_75plus[data$AGEGRP2 != ">=75"] <- 0
  
  #ECOG_C: 
  #- Ensure that all observations stay within allowed range from protocol (0-2)
  #- Define new dummy coded variables with reference group ECOG = 0.
  data$ECOG_C[data$ECOG_C == 3] <- NA
  data$ECOG_1 <- NA
  data$ECOG_1[data$ECOG_C == 1] <- 1
  data$ECOG_1[data$ECOG_C != 1] <- 0
  data$ECOG_2 <- NA
  data$ECOG_2[data$ECOG_C == 2] <- 1
  data$ECOG_2[data$ECOG_C != 2] <- 0
  
  
  #Construct outcomes
    #Died within 2 years of study commencement:
    data$DEATH2YRS <- 0
    data$DEATH2YRS[data$DEATH == "YES" & data$LKADT_P < 365*2] <- 1
    
    #Discontinued treatment within 3 months: recode NAs to 0 (no) 
    data$DISCONT[is.na(data$DISCONT)] <- 0
  
    
  #Drop variables that we will not be using because we will use their new definitions from
  #above
  data <- data[, setdiff(names(data), c("RACE_C", "AGEGRP2", "AGEGRP", 
                                        "ECOG_C", "DEATH", "LKADT_P"))]
  
  #Drop variables that we will not be using because they are empty (NA for all) or because they
  #were not available in the DREAM Challenge test data set. 
  data <- data[, setdiff(names(data), c("LKADT_PER", "HGTBLCAT", "WGTBLCAT", "TRT1_ID",
                                        "TRT2_ID", "TRT3_ID", "HEAD_AND_NECK", "STOMACH",
                                        "PANCREAS", "THYROID", "DOMAIN", "RPT", 
                                        "ENDTRS_C", "ENTRT_PC", "PER_REF", "LKADT_REF",
                                        "LKADT_PER", "SMOKE", "SMOKSTAT", "SMOKFREQ",
                                        "GLEAS_DX", "WEIGHTBL", "TSTAG_DX", "STUDYID"))]
  
  #Drop variables with >10% missing information
  missVar <- sapply(data, function(x) mean(is.na(x)) <= 0.1)
  data <- data[, unlist(missVar)]

  #drop observations with missing information (around 150 people)
  data <- na.omit(data)
  
  #Rearrange variables so that outcomes are first
  data <- data[, c("DEATH2YRS", "DISCONT", setdiff(names(data), c("DEATH2YRS", "DISCONT")))]
  
  #define test and train data
  set.seed(18123)
  testIndexes <- sample(1:nrow(data), size = floor(0.2*nrow(data)), replace = TRUE)
  testdata <- data[testIndexes, ]
  traindata <- data[-testIndexes, ]
  
  
  #Shuffle observations randomly in both train and test data
  set.seed(241)
  traindata <- traindata[sample(1:nrow(traindata), size = nrow(traindata)),]
  testdata <- testdata[sample(1:nrow(testdata), size = nrow(testdata)),]
  
  #Load data dictionary and add labels and descriptions to variables
  #that we have kept from the raw dataset in traindata
  info <- read.csv("./data/Coretable_labels.csv", header = TRUE, 
                   stringsAsFactors = FALSE)
  info$Var.Name[is.na(info$Var.Name)] <- "NA."
  for (i in 1:ncol(traindata)) {
    thisName <- names(traindata)[i]
    if (thisName %in% info$Var.Name) {
      attr(traindata[, thisName], "label") <- info[info$Var.Name == thisName, "Label"]
      thisInfo <- info[info$Var.Name == thisName, "Comment"]
      thisInfo <- gsub("YES", "1", thisInfo)
      thisInfo <- gsub("Y ", "1 ", thisInfo)
      thisInfo <- gsub("missing", "0", thisInfo)
      thisInfo <- gsub("No", "not", thisInfo)
      attr(traindata[, thisName], "shortDescription") <- thisInfo
    }
  }
  
  #Add labels and descriptions to variables that we have constructed in traindata
  attr(traindata$RACE_nonwhite, "label") <- 
    "Dummy coded race variable (non-white)"
  attr(traindata$RACE_nonwhite, "shortDescription") <- 
    "Categorical race variable, \'white\' is reference group."
  attr(traindata$AGEGRP_65to74, "shortDescription") <- "1 if the person is 65-74 years old at baseline, otherwise 0."
  attr(traindata$AGEGRP_75plus, "shortDescription") <- "1 if the person is >=75 years old at baseline, otherwise 0."
  attr(traindata$AGEGRP_65to74, "label") <- "Dummy coded age group variable (65-74)." 
  attr(traindata$AGEGRP_75plus, "label") <- "Dummy coded age group variable (+75)"
  attr(traindata$ECOG_1, "shortDescription") <- "1 if the person has ECOG C = 1, 0 otherwise."
  attr(traindata$ECOG_2, "shortDescription") <- "1 if the person has ECOG C = 2, 0 otherwise."
  attr(traindata$ECOG_1, "label") <- "Dummy coded ECOG C variable (1)"
  attr(traindata$ECOG_2, "label") <- "Dummy coded ECOG C variable (2)"
  attr(traindata$DISCONT, "shortDescription") <- 
    "Indicator of whether the patient discontinued treatment due to adverse effects (1) within the first 3 months of the study" 
  attr(traindata$DISCONT, "label") <- "Discontinuation indicator" 
  attr(traindata$DEATH2YRS, "shortDescription") <- 
    "Indicator of whether the patient was registered as dead (1) within the first 2 years of the study"
  attr(traindata$DEATH2YRS, "label") <- "Death indicator"
  
  
  #Split train and test data into x (features) and y datasets (outcomes) and save all of it:
  traindata_x <- traindata[, setdiff(names(traindata), c("DEATH2YRS", "DISCONT"))]
  testdata_x <- testdata[, setdiff(names(testdata), c("DEATH2YRS", "DISCONT"))]
  traindata_DEATH2YRS <- traindata$DEATH2YRS
  testdata_DEATH2YRS <- testdata$DEATH2YRS
  traindata_DISCONT <- traindata$DISCONT
  testdata_DISCONT <- testdata$DISCONT
  
  rownames(traindata_x) <- NULL
  rownames(testdata_x) <- NULL
  
  save(list = c("traindata_x", "testdata_x",
                "traindata_DEATH2YRS", "testdata_DEATH2YRS",
                "traindata_DISCONT", "testdata_DISCONT"),
       file = "./data/andata.rda")
  
  #make codebook
  makeCodebook(traindata, replace = TRUE, file = "codebook_mCRPCdata.rmd",
               reportTitle = "Codebook for mCPRPC training data")
  
