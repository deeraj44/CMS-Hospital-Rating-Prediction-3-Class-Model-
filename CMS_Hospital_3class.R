
##  Load Libraries
library(tidyverse)
library(readr)
library(janitor)
library(caret)
library(randomForest)
library(smotefamily)
library(DALEX)
library(ggplot2)
library(reshape2)
library(DT)
```

## Load and Clean Data
hospital_data <- read_csv("Hospital_General_Information.csv") %>%
  clean_names() %>%
  filter(hospital_type == "Acute Care Hospitals") %>%
  mutate(
    hospital_overall_rating = na_if(hospital_overall_rating, "Not Available"),
    hospital_overall_rating = as.numeric(hospital_overall_rating),
    rating_group = case_when(
      hospital_overall_rating %in% c(1, 2) ~ "Low",
      hospital_overall_rating == 3 ~ "Medium",
      hospital_overall_rating %in% c(4, 5) ~ "High"
    ),
    rating_group = as.factor(rating_group)
  ) %>%
  drop_na(rating_group)


## Summary Statistics
glimpse(hospital_data)
summary(hospital_data)


## Average Rating by State
avg_rating_by_state <- hospital_data %>%
  group_by(state) %>%
  summarise(avg_rating = mean(hospital_overall_rating, na.rm = TRUE),
            count = n()) %>%
  arrange(desc(avg_rating))

ggplot(avg_rating_by_state, aes(x = reorder(state, avg_rating), y = avg_rating)) +
  geom_bar(stat = "identity", fill = "blue") +
  coord_flip() +
  labs(title = "Average Hospital Rating by State", x = "State", y = "Average Rating") +
  theme_minimal()


## Ratings by Ownership
ownership_ratings <- hospital_data %>%
  filter(!is.na(hospital_overall_rating)) %>%
  group_by(hospital_ownership) %>%
  summarise(avg_rating = mean(hospital_overall_rating, na.rm = TRUE),
            count = n()) %>%
  arrange(desc(avg_rating))

ggplot(ownership_ratings, aes(x = reorder(hospital_ownership, avg_rating), y = avg_rating)) +
  geom_bar(stat = "identity", fill = "darkgreen") +
  coord_flip() +
  labs(title = "Average Ratings by Ownership Type", x = "Ownership", y = "Rating") +
  theme_minimal()


## Regional Analysis
state_region_map <- list(
  Northeast = c("CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"),
  Midwest = c("IL", "IN", "IA", "KS", "MI", "MN", "MO", "NE", "ND", "OH", "SD", "WI"),
  South = c("DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV", "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX"),
  West = c("AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA")
)

get_region <- function(state) {
  for (r in names(state_region_map)) {
    if (state %in% state_region_map[[r]]) return(r)
  }
  return(NA)
}

hospital_data$region <- sapply(hospital_data$state, get_region)

region_avg <- hospital_data %>%
  group_by(region) %>%
  summarise(avg_rating = mean(hospital_overall_rating, na.rm = TRUE))

ggplot(region_avg, aes(x = reorder(region, avg_rating), y = avg_rating)) +
  geom_bar(stat = "identity", fill = "purple") +
  labs(title = "Average Rating by U.S. Region", x = "Region", y = "Average Rating") +
  theme_minimal()


## Correlation Heatmap
corr_data <- hospital_data %>%
  select(
    hospital_overall_rating,
    count_of_facility_mort_measures,
    count_of_mort_measures_worse,
    count_of_safety_measures_worse,
    count_of_facility_safety_measures,
    count_of_readm_measures_worse,
    count_of_facility_readm_measures
  ) %>%
  mutate(across(everything(), as.numeric)) %>%
  drop_na()

corr_matrix <- cor(corr_data)
melted_corr <- melt(corr_matrix)

ggplot(melted_corr, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "orange", high = "darkred", mid = "red", midpoint = 0) +
  labs(title = "Correlation Heatmap") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


## Top Rated Hospitals
top_hospitals <- hospital_data %>%
  filter(hospital_overall_rating == 5) %>%
  group_by(hospital_ownership) %>%
  slice_max(order_by = hospital_overall_rating, n = 5, with_ties = FALSE) %>%
  select(facility_name, state, hospital_ownership, hospital_overall_rating)

datatable(top_hospitals, caption = "Top 5 Hospitals by Ownership Type")



##  Feature Engineering
model_data <- hospital_data %>%
  mutate(
    readm_safety_gap = as.numeric(count_of_readm_measures_worse) - as.numeric(count_of_safety_measures_worse),
    mort_ratio = as.numeric(count_of_mort_measures_worse) / (as.numeric(count_of_facility_mort_measures) + 1),
    safety_ratio = as.numeric(count_of_safety_measures_worse) / (as.numeric(count_of_facility_safety_measures) + 1)
  ) %>%
  select(rating_group, hospital_ownership, state, readm_safety_gap, mort_ratio, safety_ratio) %>%
  mutate(across(c(hospital_ownership, state), as.factor)) %>%
  drop_na()



## Train-Test Split
set.seed(123)
train_idx <- createDataPartition(model_data$rating_group, p = 0.8, list = FALSE)
train <- model_data[train_idx, ]
test <- model_data[-train_idx, ]

##  Balance Classes with SMOTE
X <- train %>% select(-rating_group)
y <- train$rating_group

X_numeric <- data.frame(model.matrix(~ . - 1, data = X))
smote_result <- SMOTE(X_numeric, y, K = 5)

balanced_train <- smote_result$data
balanced_train$rating_group <- as.factor(smote_result$data$class)
balanced_train$class <- NULL


##  Train Random Forest Classifier
model <- train(rating_group ~ ., data = balanced_train, method = "rf")


##  Evaluation
## Preprocess test
X_test <- data.frame(model.matrix(~ . - 1, data = test %>% select(-rating_group)))
y_test <- test$rating_group

## Align test to training columns
missing_cols <- setdiff(colnames(balanced_train)[-ncol(balanced_train)], colnames(X_test))
X_test[missing_cols] <- 0
X_test <- X_test[, colnames(balanced_train)[-ncol(balanced_train)]]

## Predict
preds <- predict(model, newdata = X_test)

## Confusion matrix
confusionMatrix(preds, y_test)


##  Feature Importance
varImpPlot(model$finalModel)


##  SHAP/DALEX Model Explanation
explainer <- explain(model$finalModel, data = X_test, y = y_test, label = "Random Forest")
model_parts(explainer) %>% plot()


##  Class Distribution After SMOTE
ggplot(balanced_train, aes(x = rating_group)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Balanced Class Distribution After SMOTE")
