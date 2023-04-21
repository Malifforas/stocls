import argparse
import sys

import data
import evaluate
import model
import portfolio
import preprocessing
import train
import trader


def main(args):
    if args.mode == 'train':
        # Load data
        X_train, y_train, X_test, y_test = data.load_data(args.ticker, args.start_date, args.end_date, args.features)

        # Preprocess data
        X_train, y_train, X_test, y_test, scaler = preprocessing.preprocess_data(X_train, y_train, X_test, y_test)

        # Train model
        history, trained_model = train.train_model(args.model_type, X_train, y_train, X_test, y_test, args.batch_size, args.epochs)

        # Evaluate model
        evaluate.evaluate_model(trained_model, X_test, y_test, scaler)

        # Save model
        trained_model.save_model(args.model_output_path)

    elif args.mode == 'trade':
        # Load model
        trained_model = model.load_model(args.model_path)

        # Load data
        X, y = data.load_live_data(args.ticker, args.interval, args.features)

        # Preprocess data
        X, y, scaler = preprocessing.preprocess_live_data(X, y)

        # Make predictions
        predictions = trained_model.predict(X)

        # Get current stock price
        current_price = trader.get_current_price(args.ticker)

        # Calculate trade signal
        trade_signal = trader.calculate_trade_signal(predictions[-1], current_price, args.investment_amount)

        # Execute trade
        if trade_signal:
            portfolio.execute_trade(args.ticker, args.interval, args.investment_amount, trade_signal)
        else:
            print("No trade signal generated.")

    else:
        print("Invalid mode specified.")
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stock Trader')
    parser.add_argument('mode', choices=['train', 'trade'], help='Mode of operation.')
    parser.add_argument('--ticker', required=True, help='Ticker symbol of the stock to trade.')
    parser.add_argument('--start_date', help='Start date for historical data.')
    parser.add_argument('--end_date', help='End date for historical data.')
    parser.add_argument('--features', nargs='+', default=['open', 'high', 'low', 'close', 'volume'], help='Features to use.')
    parser.add_argument('--model_type', choices=['lstm', 'cnn'], default='lstm', help='Type of model to use.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training.')
    parser.add_argument('--model_output_path', help='Path to save trained model.')
    parser.add_argument('--model_path', help='Path to trained model.')
    parser.add_argument('--interval', choices=['1m', '5m', '15m', '30m', '60m'], default='1m', help='Interval for live data.')
    parser.add_argument('--investment_amount', type=float, default=1000.0, help='Amount to invest in each trade.')
    args = parser.parse_args()

    main(args)