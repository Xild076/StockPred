# StockPred — A novel approach at predicting stock prices
StockPred presents a novel approach at predicting stock prices. It implements a lightweight architecture inspired by LLMs to achieve surprisingly high accuracy.
---
### Efficiency
The model is only has about 2M parameters, making it very efficient and runnable on low-end GPUs and high-end CPUs. This makes it one of the most accessible stock prediction models out there, especially given its accuracy.
### Accuracy
The latest trained model, 2025-07-06T13:00:17.372776, when tested over 3562 samples, returned:
- A directional accuracy of 87.20%
- A trading winrate of 86.69%
- A median R^2 score of 0.8193
- A positive R^2 % rate of 93.68%
All of these scores are well above the average for models of this size and the directional accuracy is also well above even larger models.
### Limitations
However, the models come at limitations too:
- A R^2 score of -1.8326
- A RMSE of 140.0130
- A MAPE of 242.48%
Despite having overall good accuracy, there are a few disasterously horrible predictions, leading to horrible predictions shown above. 
## Usage
You can use this model at:
- https://xild-stockpred.streamlit.app/
