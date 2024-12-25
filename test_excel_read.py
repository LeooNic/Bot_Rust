from market_data_processor import MarketData
import os

def test_excel_reading():
    market_data = MarketData()
    
    # Ruta al archivo Excel
    excel_path = "BTCUSDT_2021.xlsx"  # Asegúrate de que este es el nombre correcto
    sheet_name = "1 hora"  # Asegúrate de que este es el nombre correcto de la hoja
    
    print(f"Intentando leer: {excel_path}, hoja: {sheet_name}")
    
    try:
        # Leer el Excel
        market_data.read_excel(excel_path, sheet_name)
        
        # Obtener los datos
        data_str = market_data.get_data_as_str(sheet_name)
        
        print("Primeras filas del DataFrame:")
        print(data_str)
        
        print("\nLectura exitosa!")
        
    except Exception as e:
        print(f"Error al leer el Excel: {str(e)}")

if __name__ == "__main__":
    test_excel_reading()