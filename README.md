## Рекомендательная система для Андроид приложения. 
1. Для построения рекомендательной системы были опробованы два метода: __collaborative filtering__, __content based recommendations__.
2. Так как данные о взаимодействиях пользователей с товарами отсутствуют, были сгенерированы данные для коллаборативной фильтрации (Generation_synthetic_data.ipynb)
3. В файле Recomendations_with_db.py прописано взаимодействие с базой данных для получения данных о товарах, просмотренных пользователем, и получение рекомендаций для данного пользователя.
