library(shiny)
library(randomForest)
rf_common_model <- readRDS("random_forest_model.rds")
data <- read.csv("fraud_oracle2.csv")

categorical_vars <- c("Make", "MonthClaimed", "WeekOfMonthClaimed", "Sex", "MaritalStatus", 
                      "Fault", "PolicyType", "VehicleCategory", "VehiclePrice", "PastNumberOfClaims")
data[categorical_vars] <- lapply(data[categorical_vars], as.factor)
ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      /* Custom CSS for shiny app */
      .main-container {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
      }
      .btn-primary {
        background-color: #007bff !important;
        border-color: #007bff !important;
        color: #fff !important;
        font-weight: bold !important;
      }
      .btn-primary:hover {
        background-color: #0056b3 !important;
        border-color: #0056b3 !important;
      }
      .btn-primary:focus {
        box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.5) !important;
      }
      .form-group > label {
        color: #007bff !important;
        font-weight: bold !important;
      }
      .form-group > .form-control {
        border-color: #007bff !important;
      }
      .form-group > .form-control:focus {
        box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.5) !important;
        border-color: #007bff !important;
      }
    "))
  ),
  titlePanel("Vehicle Insurance Claim Fraud Detection"),
  div(class = "main-container",
      sidebarLayout(
        sidebarPanel(
          # Input fields for each feature
          selectInput("make", "Make", choices = c("", "Accura", "BMW", "Chevrolet", "Dodge", "Ferrari", "Ford", "Honda", "Jaguar", "Lexus", "Mazda", "Mecedes", "Mercury", "Nissan", "Pontiac", "Porche", "Saab", "Saturn", "Toyota", "VW")), # Update with actual categories
          fluidRow(
            column(6, selectInput("month_claimed", "Month Claimed", choices = c("", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))),
            column(6, selectInput("week_of_month_claimed", "Week of Month Claimed", choices = c("", "1", "2", "3", "4", "5")))
          ), # Update with actual categories
          
          fluidRow(
            column(6, selectInput("sex", "Sex", choices = c("", "Male", "Female"))),
            column(6, selectInput("marital_status", "Marital Status", choices = c("", "Single", "Married", "Divorced", "Widowed")))
          ),
          fluidRow(
            column(6, numericInput("age", "Age", value = NULL, min = 0, max = 100)),
            column(6, numericInput("rep_number", "Rep Number", value = NULL, min = 0))
          ),
          fluidRow(
            column(6, numericInput("deductible", "Deductible", value = NULL, min = 0, max = 1000)),
            column(6, numericInput("driver_rating", "Driver Rating", value = NULL, min = 0, max = 5))
          ),fluidRow(
            column(6, selectInput("fault", "Fault", choices = c("", "Policy Holder", "Third Party"))),
            column(6, selectInput("policy_type", "Policy Type", choices = c("", "Sedan - All Perils", "Sport - All Perils", "Sedan - Collision", "Sport - Collision", "Utility - All Perils", "Utility - Collision", "Sport - Liability", "Sedan - Liability"))),
          ),
          fluidRow(
            column(6, selectInput("vehicle_category", "Vehicle Category", choices = c("", "Sedan", "Sport", "Utility"))),
            column(6, selectInput("vehicle_price", "Vehicle Price", choices = c("", "20000 to 29000", "30000 to 39000", "40000 to 59000", "60000 to 69000", "more than 69000")))
          ),
          fluidRow(
            column(6,selectInput("past_number_of_claims", "Past Number of Claims", choices = c("", "1", "2 to 4", "more than 4","none"))), # Update with actual categories 
          ),
          
          actionButton("predict_btn", "Predict Fraud", class = "btn-primary")
        ),
        mainPanel(
          tabsetPanel(
            tabPanel("Prediction Result", textOutput("prediction_result"))
          )
        )
      )
  )
)

# Define server logic
server <- function(input, output) {
  observeEvent(input$predict_btn, {
    # Prepare input data
    input_data <- data.frame(
      Make = factor(input$make, levels = levels(data$Make)),
      MonthClaimed = factor(input$month_claimed, levels = levels(data$MonthClaimed)),
      WeekOfMonthClaimed = factor(input$week_of_month_claimed, levels = levels(data$WeekOfMonthClaimed)),
      Sex = factor(input$sex, levels = levels(data$Sex)),
      MaritalStatus = factor(input$marital_status, levels = levels(data$MaritalStatus)),
      Age = input$age,
      Fault = factor(input$fault, levels = levels(data$Fault)),
      PolicyType = factor(input$policy_type, levels = levels(data$PolicyType)),
      VehicleCategory = factor(input$vehicle_category, levels = levels(data$VehicleCategory)),
      VehiclePrice = factor(input$vehicle_price, levels = levels(data$VehiclePrice)),
      RepNumber = input$rep_number,
      Deductible = input$deductible,
      DriverRating = input$driver_rating,
      PastNumberOfClaims = factor(input$past_number_of_claims, levels = levels(data$PastNumberOfClaims))
    )
    
    # Check for missing values
    if (anyNA(input_data)) {
      # Display error message if there are missing values
      output$prediction_result <- renderText({
        "Error: Missing values detected in input data"
      })
    } else {
      # Make prediction using the rf_common_model
      prediction <- predict(rf_common_model, newdata = input_data)
      
      # Display the prediction result
      output$prediction_result <- renderText({
        if (is.na(prediction)) {
          "Error: Unable to make prediction"
        } else if (prediction == 1) {
          "Fraud Found"
        } else {
          "No Fraud Found"
        }
      })
    }
  })
}

# Run the Shiny app
shinyApp(ui = ui, server = server)