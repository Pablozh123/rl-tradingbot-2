import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path so we can import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_loader import DataLoader

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        # Create a small dummy dataframe
        dates = pd.date_range(start='2023-01-01', periods=100, freq='5min')
        self.df = pd.DataFrame({
            'timestamp': dates,
            'Open': np.linspace(100, 110, 100),
            'High': np.linspace(101, 111, 100),
            'Low': np.linspace(99, 109, 100),
            'Close': np.linspace(100, 110, 100), # Perfect uptrend
            'Volume': np.random.rand(100) * 100
        })
        
        # Add some volatility for ATR calculation
        self.df.loc[50:60, 'Close'] = self.df.loc[50:60, 'Close'] + np.random.normal(0, 0.1, 11)

    def test_grind_detection(self):
        """Test if the grind logic detects a perfect linear trend."""
        # Save dummy data to temp file
        self.df.to_csv('tests/temp_test_data.csv', index=False)
        
        loader = DataLoader('tests/temp_test_data.csv', window_size=20)
        # Run the full preprocessing pipeline
        df = loader.preprocess()
        
        # Check if feature columns exist
        self.assertIn('trend_slope', df.columns)
        self.assertIn('trend_r2', df.columns)
        self.assertIn('volatility_ratio', df.columns)
        
        # Verify R2 for the linear segment (first 50 candles are linear)
        # We check index 30 (window is 20, so it has enough data)
        # The trend is perfect, so R2 should be very close to 1.0
        r2_at_30 = df['trend_r2'].iloc[30]
        self.assertGreater(r2_at_30, 0.99, "R2 should be > 0.99 for linear data")
        
        # Verify Slope
        # Our data goes from 100 to 110 over 100 steps. Slope approx 0.1
        slope_at_30 = df['trend_slope'].iloc[30]
        self.assertAlmostEqual(slope_at_30, 0.1, delta=0.01, msg="Slope should be approx 0.1")

        # Clean up
        if os.path.exists('tests/temp_test_data.csv'):
            os.remove('tests/temp_test_data.csv')

    def test_sfp_detection(self):
        """Test Swing Failure Pattern (SFP) logic."""
        # We mock _calculate_4h_structure to return known pivot levels
        # This isolates the SFP logic from the pivot finding logic
        
        # Create a mock 4H structure dataframe
        # It needs to cover the timestamp of our test data
        mock_4h_data = pd.DataFrame({
            'active_pivot_high': [105.0], # Pivot High at 105
            'active_pivot_low': [95.0],   # Pivot Low at 95
            'bullish_range_low': [np.nan],
            'bullish_range_high': [np.nan],
            'bearish_range_low': [np.nan],
            'bearish_range_high': [np.nan]
        }, index=[self.df['timestamp'].iloc[0]]) # Valid from start
        
        # Modify test data to trigger SFP
        # Bearish SFP: High > 105, Close < 105
        self.df.loc[10, 'High'] = 106.0
        self.df.loc[10, 'Close'] = 104.0
        
        # Bullish SFP: Low < 95, Close > 95
        self.df.loc[20, 'Low'] = 94.0
        self.df.loc[20, 'Close'] = 96.0
        
        self.df.to_csv('tests/temp_test_sfp.csv', index=False)
        
        with patch.object(DataLoader, '_calculate_4h_structure', return_value=mock_4h_data):
            loader = DataLoader('tests/temp_test_sfp.csv', window_size=20)
            df = loader.preprocess()
            
            # Check Bearish SFP at index 10
            self.assertTrue(df['is_sfp_short'].iloc[10], "Should detect Bearish SFP")
            
            # Check Bullish SFP at index 20
            self.assertTrue(df['is_sfp_long'].iloc[20], "Should detect Bullish SFP")
            
            # Check False Positive (Normal candle)
            self.assertFalse(df['is_sfp_short'].iloc[11], "Should not detect SFP on normal candle")

        if os.path.exists('tests/temp_test_sfp.csv'):
            os.remove('tests/temp_test_sfp.csv')

    def test_gp_detection(self):
        """Test Golden Pocket (GP) distance logic."""
        # Mock GP Zones
        mock_4h_data = pd.DataFrame({
            'active_pivot_high': [np.nan],
            'active_pivot_low': [np.nan],
            'bullish_range_low': [90.0],   # GP Zone 90-92
            'bullish_range_high': [92.0],
            'bearish_range_low': [108.0],  # GP Zone 108-110
            'bearish_range_high': [110.0]
        }, index=[self.df['timestamp'].iloc[0]])
        
        # Modify test data
        # 1. Inside Bullish GP
        self.df.loc[10, 'Low'] = 91.0 # Inside 90-92
        
        # 2. Near Bullish GP
        self.df.loc[11, 'Low'] = 93.0 # 1 unit away from 92
        
        # 3. Inside Bearish GP
        self.df.loc[20, 'High'] = 109.0 # Inside 108-110
        
        self.df.to_csv('tests/temp_test_gp.csv', index=False)
        
        with patch.object(DataLoader, '_calculate_4h_structure', return_value=mock_4h_data):
            loader = DataLoader('tests/temp_test_gp.csv', window_size=20)
            df = loader.preprocess()
            
            # Check Distance inside Bullish GP (Should be 0)
            self.assertEqual(df['dist_to_gp_long'].iloc[10], 0.0, "Distance inside GP should be 0")
            
            # Check Distance near Bullish GP
            # Dist = |93 - 92| = 1. Normalized by Close (approx 100) -> 0.01
            # We just check it's positive and roughly correct
            self.assertGreater(df['dist_to_gp_long'].iloc[11], 0.0)
            self.assertLess(df['dist_to_gp_long'].iloc[11], 0.1)
            
            # Check Distance inside Bearish GP
            self.assertEqual(df['dist_to_gp_short'].iloc[20], 0.0, "Distance inside Bearish GP should be 0")

        if os.path.exists('tests/temp_test_gp.csv'):
            os.remove('tests/temp_test_gp.csv')


if __name__ == '__main__':
    unittest.main()
