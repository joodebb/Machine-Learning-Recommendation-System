E-Commerce  Machine Learning Product Recommendation System Using Customer Interaction and Purchase History

         1. Project Overview
This project involved developing a machine learning-based recommendation system for an e-commerce platform using customer online interaction data and purchase history. The research explored how customer engagement and purchasing behaviour could be leveraged to generate personalised product recommendations, improve customer experience and support business sales.
The solution implemented a Memory-Based Collaborative Filtering approach using the K-Nearest Neighbours (KNN) algorithm to identify similarities between customers and generate relevant product suggestions.


2. Aims & Objectives
Project Aim
To design and implement an intelligent recommendation engine that provides personalised product suggestions based on customer interaction patterns and purchasing behaviour.
Project Objectives
•	Collect and explore customer interaction and purchase history data.
•	Perform data preprocessing, including handling missing values and normalising numerical features.
•	Analyse relationships between customer engagement metrics and spending behaviour using statistical analysis and data visualisations.
•	Develop a recommendation model using Memory-Based Collaborative Filtering and the K-Nearest Neighbours algorithm.
•	Identify the optimal number of nearest neighbours through cross-validation.
•	Implement a recommendation engine to generate personalised product suggestions.
•	Evaluate recommendation performance using Mean Average Precision (MAP).
3. Methods & Technical Implementation
The project followed a structured machine learning workflow:
Data Collection and Exploration
•	Customer interaction data, including time spent on the website and mobile application.
•	Purchase behaviour, including yearly spending and customer membership duration.
•	Exploratory Data Analysis (EDA) using summary statistics, scatter plots and correlation analysis.
Data Preprocessing
•	Identified and handled missing values.
•	Normalised numerical features to ensure consistency across different measurement scales.
•	Selected four key customer attributes:
o	Time on App
o	Time on Website
o	Length of Membership
o	Yearly Amount Spent
Model Development
•	Implemented a Memory-Based Collaborative Filtering recommendation approach.
•	Applied the K-Nearest Neighbours (KNN) algorithm to identify similar customers.
•	Tested different K values ranging from 1 to 20 using cross-validation.
•	Selected K = 5 based on the highest cross-validation accuracy.
•	Developed a recommendation function to generate product suggestions.
Tools & Technologies
•	Python
•	Pandas
•	NumPy
•	Scikit-learn
•	Matplotlib
•	Seaborn
•	Jupyter Notebook
•	Machine Learning
•	Collaborative Filtering
•	K-Nearest Neighbours (KNN)
4. Evaluation & Results
The recommendation model was evaluated using Mean Average Precision (MAP), a ranking-based metric that measures how effectively a recommendation system identifies and ranks relevant items.
Key Findings from Exploratory Data Analysis
•	Time on App vs Yearly Amount Spent: A positive covariance of 39.82 indicated an association between increased mobile application engagement and higher customer spending.
•	Time on Website vs Yearly Amount Spent: A negative covariance of -0.28 suggested a very weak negative association between website engagement and yearly spending.
•	Length of Membership vs Yearly Amount Spent: A positive covariance of 64.123 indicated that longer-standing customers tended to spend more.
Model Evaluation
•	Tested K values between 1 and 20.
•	Used cross-validation to identify the optimal K value.
•	Selected K = 5 for the final recommendation model.
•	The reported MAP evaluation score was 1.0, indicating perfect ranking performance on the evaluated test data.
5. Conclusions & Recommendations
The research demonstrated how customer engagement and purchasing behaviour can be used to develop personalised recommendation systems for e-commerce platforms.
The key conclusions were:
•	The KNN algorithm provided a practical approach to identifying similar customers and generating product recommendations.
•	Customer engagement on mobile applications and membership duration showed positive associations with yearly spending.
•	Cross-validation provided a systematic approach to selecting the optimal number of neighbours, with K = 5 producing the best cross-validation accuracy.
•	The recommendation engine demonstrated strong performance on the evaluated dataset.
Limitations and Future Improvements
One limitation was the absence of customer demographic information, particularly gender, which occasionally resulted in recommendations that were not aligned with customer preferences.
Future improvements could include:
•	Incorporating demographic and product-category information.
•	Exploring hybrid recommendation techniques combining collaborative and content-based filtering.
•	Applying deep learning approaches to improve recommendation personalisation.
•	Testing the model on larger and more diverse datasets.
•	Evaluating generalisation using additional ranking metrics and unseen customer data.
6. Acknowledgements
I dedicate this research to Almighty God for His blessings and guidance throughout my academic journey.
I am deeply grateful to my family, particularly my mother, Alice, for her unconditional love, encouragement and unwavering support.
My sincere appreciation goes to my supervisor, Dr John N. Brown, for his invaluable mentorship, constructive feedback and guidance throughout this research.
Finally, I extend my gratitude to A.K. for their encouragement, presence and support throughout the journey.

