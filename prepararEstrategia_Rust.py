from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from market_data_processor import MarketData
import logging
from pathlib import Path
from io import StringIO
import sys
from openpyxl.utils import get_column_letter
import multiprocessing

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy_preparation.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def procesar_archivo(ruta_entrada: str, ruta_salida: str, temporalidad: str = "1 hora"):
    """
    Procesa un archivo Excel aplicando indicadores técnicos usando la implementación en Rust.
    """
    try:
        logger.info(f"Procesando archivo: {ruta_entrada}")
        
        # Leer archivo original para mantener los tipos de datos de tiempo
        df_original = pd.read_excel(ruta_entrada, sheet_name=temporalidad)
        logger.info(f"Columnas en archivo original: {df_original.columns.tolist()}")
        
        # Procesar con MarketData
        market_data = MarketData()
        market_data.read_excel(ruta_entrada, temporalidad)
        
        market_data.apply_all_indicators(
            sheet_name=temporalidad,
            smoothing_period=21,
            adx_period=14,
            rsi_period=14
        )
        
        # Obtener datos y convertir a DataFrame
        df_str = market_data.get_data_as_str(temporalidad)
        
        try:
            # Configurar los tipos de datos para la lectura
            dtype_dict = {
                'open': 'float64',
                'high': 'float64',
                'low': 'float64',
                'close': 'float64',
                'volume': 'float64',
                'adx': 'float64',
                'DIPlus': 'float64',
                'DIMinus': 'float64',
                'rsi': 'float64',
                'atr': 'float64',
                'ema100': 'float64',
                'ema9': 'float64',
                'ema20': 'float64',
                'Trend': 'float64',
                'Color': 'str'
            }
            
            # Leer CSV con tipos de datos específicos
            df = pd.read_csv(
                StringIO(df_str),
                dtype=dtype_dict,
                parse_dates=['time']
            )
            
            # Restaurar la columna time del DataFrame original
            df['time'] = df_original['time']
            
            logger.info(f"Tipos de datos después de la conversión:\n{df.dtypes}")
            
            # Guardar resultado con formato mejorado
            with pd.ExcelWriter(ruta_salida, engine='openpyxl') as writer:
                df.to_excel(
                    writer,
                    sheet_name=temporalidad,
                    index=False,
                    float_format='%.6f'  # Formato para números flotantes
                )
                
                # Ajustar el formato de las columnas
                worksheet = writer.sheets[temporalidad]
                for idx, col in enumerate(df.columns):
                    column_letter = get_column_letter(idx + 1)
                    
                    if col == 'time':
                        worksheet.column_dimensions[column_letter].width = 20
                    elif col == 'Color':
                        worksheet.column_dimensions[column_letter].width = 10
                    else:
                        worksheet.column_dimensions[column_letter].width = 15
                        
            logger.info(f"Archivo procesado exitosamente: {ruta_salida}")
            return True, ruta_entrada
            
        except Exception as e:
            logger.error(f"Error procesando DataFrame: {str(e)}")
            return False, ruta_entrada
        
    except Exception as e:
        logger.error(f"Error procesando archivo {ruta_entrada}: {str(e)}")
        return False, ruta_entrada

def procesar_archivos_paralelo(rutas_entrada, rutas_salida, num_workers=None):
    """
    Procesa múltiples archivos en paralelo usando ProcessPoolExecutor.
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    logger.info(f"Iniciando procesamiento en paralelo con {num_workers} workers")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Crear lista de futures
        futures = [
            executor.submit(procesar_archivo, entrada, salida)
            for entrada, salida in zip(rutas_entrada, rutas_salida)
        ]
        
        # Procesar resultados a medida que se completan
        for future in as_completed(futures):
            try:
                success, ruta = future.result()
                if success:
                    logger.info(f"Procesamiento completado para {ruta}")
                else:
                    logger.error(f"Error procesando {ruta}")
            except Exception as e:
                logger.error(f"Error inesperado: {str(e)}")

def main():
    # Configuración
    par = "BTCUSDT"
    años = ["2021", "2022", "2023", "2024"]
    
    # Preparar rutas
    rutas_entrada = [
        str(Path(f"Data/{par}/{par}_{año}.xlsx"))
        for año in años
    ]
    
    rutas_salida = [
        str(Path(f"Data/{par}/Señales{año}Estrategia5{par}.xlsx"))
        for año in años
    ]
    
    try:
        logger.info("Iniciando procesamiento de archivos")
        logger.info(f"Rutas de entrada: {rutas_entrada}")
        logger.info(f"Rutas de salida: {rutas_salida}")
        
        # Procesar archivos en paralelo
        procesar_archivos_paralelo(rutas_entrada, rutas_salida)
        
        logger.info("Procesamiento completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error en la ejecución principal: {str(e)}")

if __name__ == "__main__":
    main()