# used_car_price_prediction
This project aims to predict used car prices using features like Brand, Model, Vehicle Age, KM Driven, Seller Type, Fuel Type, Transmission Type, Mileage, Engine, Max Power, and Seats. Various machine learning models were employed to achieve the best prediction accuracy.

Data Transformation:

Handling Missing Values:

Missing data was filled using techniques like mean, median, or mode based on the feature type.

Feature Encoding:

Categorical features (e.g., Fuel Type, Transmission) were encoded into numerical values using techniques like One-Hot Encoding and Label Encoding.

Feature Scaling:

Numerical features were scaled using StandardScaler for uniformity.

Derived Features:

Vehicle Age was calculated as a derived feature from the year of manufacture.

Model Evaluation:

Models like Linear Regression, Random Forest, Gradient Boosting, and XGBoost were trained and evaluated.
Metrics like R² Score and Mean Absolute Error (MAE) were used to identify the best-performing model.
The project successfully built a model capable of accurately predicting used car prices, providing valuable insights for buyers and sellers.
