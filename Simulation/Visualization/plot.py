import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/mnt/data/home/zayd/Digital_twin_project/machine_learning/dataset/new_dataset.csv')

# Plot predictions
plt.figure(figsize=(14, 7))
# plt.plot(dates_train, target_scaler.inverse_transform(y_train_lstm.reshape(-1, 1)).flatten(), label='Training Data')
plt.plot(df['time_id'], df['J311'], label='Actual', color='orange')
#plt.plot(test_scenarios, y_pred_inv, label='Predicted', color='green')
plt.xlabel('Date')
plt.ylabel('Water Consumption')
plt.title('LSTM Model Forecast vs Actual')
plt.legend()
plt.show()
