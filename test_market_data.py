import unittest
import pandas as pd
import numpy as np
from market_data_processor import MarketData

class TestMarketDataProcessor(unittest.TestCase):
    def setUp(self):
        self.market_data = MarketData()
        
    def test_read_excel(self):
        """Test la lectura de archivos Excel."""
        try:
            self.market_data.read_excel("test_data.xlsx", "1 hora")
            data_str = self.market_data.get_data_as_str("1 hora")
            self.assertIsNotNone(data_str)
        except Exception as e:
            self.fail(f"read_excel raised {type(e).__name__} unexpectedly!")
            
    def test_indicators_calculation(self):
        """Test el cálculo de indicadores."""
        try:
            # Cargar datos de prueba
            self.market_data.read_excel("test_data.xlsx", "1 hora")
            
            # Aplicar indicadores
            self.market_data.apply_all_indicators(
                sheet_name="1 hora",
                smoothing_period=21,
                adx_period=14,
                rsi_period=14
            )
            
            # Verificar resultados
            data_str = self.market_data.get_data_as_str("1 hora")
            df = pd.read_csv(pd.StringIO(data_str), sep=",")
            
            # Verificar que todos los indicadores estén presentes
            required_columns = [
                "adx", "DIPlus", "DIMinus", "rsi", "atr",
                "ema100", "ema9", "ema20", "Trend", "Color"
            ]
            
            for col in required_columns:
                self.assertIn(col, df.columns)
                
            # Verificar que no hay valores NaN
            for col in required_columns:
                self.assertFalse(df[col].isna().any())
                
        except Exception as e:
            self.fail(f"indicator calculation test failed: {str(e)}")
            
if __name__ == '__main__':
    unittest.main()