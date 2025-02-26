# ReinforceTrade Project ![ReinforceTrade Overview](screenshot.png)

This project is an algorithmic trading bot developed to simulate and analyze trading strategies in financial markets. It was built to showcase skills, and to demonstrate capabilities using machine learning techniques and data visualizations.

## Project Structure

- **Root Files**
  - `callback.py`: Functions and callbacks for interacting with the trading environment.
  - `train.py`: Script for training the trading model.
  - `tuning.py`: Script for hyperparameter tuning.
  - `download.py`: Script for downloading and preprocessing market data.
  - `test.py`: Script for testing trading model.
  - `requirements.txt`: List of dependencies required to run the project.

- **Main Directories**
  - **config/**
    - `config.py`: Global project configuration.
  - **data/**
    - Stores market data and generated datasets.
  - **env/**
    - Contains the virtual environment and related scripts (e.g., activation scripts).
  - **environment/crypto_trade/**
    - `portfolio.py`: Manages portfolio simulation, trade executions, and transaction tracking.
    - `plot_history.py`: Implements visualization of trading history and agent observations using [matplotlib](https://matplotlib.org/) and [pandas](https://pandas.pydata.org/).  
      See [plot_history.py] for details.

- **Other Directories**
  - **check_freq/** and **check_freq_optuna/**: Contains scripts/resources for execution frequency checks and performance optimization.
  - **modelos/** and **net/**: Pre-trained models and neural network configurations.
  - **ppo_tensorboard/**: Contains TensorBoard logs for visualizing the training process.

## Key Features

- **Operation and Performance Visualization**
  - The `Visualize` class in [plot_history.py] plots:
    - Midpoint prices along with trade executions (buy and sell signals).
    - Inventory levels at each step.
    - Realized Profit and Loss (PnL).
  - It also saves the agent's historical observations for further analysis.

- **Portfolio Simulation**
  - The module [portfolio.py] handles trading operations, including opening and closing positions, updating interest, and generating reports.

- **Reinforcement Learning with Stable Baselines3**
  - The project leverages **Stable Baselines3** for building and training the reinforcement learning models that power the trading agent.


## Usage Instructions

1. **Installation**
   - Ensure you have Python 3.10 or newer.
   - Install the project dependencies by running:
     ```sh
     pip install -r requirements.txt
     ```

2. **Training**
   - Run the training script:
     ```sh
     python train.py
     ```

3. **Visualizing Trading History**
   - To generate plots of a trading session, run:
     ```sh
     python plot_history.py
     ```
     This will create graphs displaying the evolution of the portfolio and executed trades.

4. **TensorBoard Integration**
   - To view detailed training metrics, launch TensorBoard by running:
     ```sh
     tensorboard --logdir=ppo_tensorboard
     ```
   - Then open [http://localhost:6006](http://localhost:6006) in your browser.

## Additional Notes

- This project leverages popular libraries such as [matplotlib](https://matplotlib.org/), [numpy](https://numpy.org/), and [pandas](https://pandas.pydata.org/) for data handling and visualization.
- **Stable Baselines3** is used for reinforcement learning, providing state-of-the-art algorithms to train the trading agent.


## Conclusion

The ReinforceTrade project demonstrates a comprehensive approach to algorithmic trading with advanced simulation and analysis capabilities. Its visualizations of portfolio evolution and transaction history make it an excellent showcase of technical proficiency in building trading applications.



---