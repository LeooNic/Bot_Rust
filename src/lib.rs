use pyo3::prelude::*;
use std::collections::HashMap;
use calamine::{Reader, open_workbook_auto, DataType};
use rayon::prelude::*;
use polars::prelude::*;
use polars::io::prelude::*;


// Helper function para convertir PolarsError a PyErr
fn convert_polars_err(err: PolarsError) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Polars error: {}", err))
}

// Helper function para convertir Series a Vec<f64>
fn series_to_vec(series: &Series) -> PyResult<Vec<f64>> {
    series
        .f64()
        .map_err(convert_polars_err)?
        .into_iter()
        .map(|opt_val| Ok(opt_val.unwrap_or(0.0)))
        .collect()
}

#[pyclass]
struct MarketData {
    data: HashMap<String, DataFrame>,
}

#[pymethods]
impl MarketData {
    #[new]
    fn new() -> Self {
        MarketData {
            data: HashMap::new(),
        }
    }

    fn read_excel(&mut self, py: Python, path: &str, sheet_name: &str) -> PyResult<()> {
        py.allow_threads(|| {
            let mut workbook = open_workbook_auto(path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
            
            let range = workbook.worksheet_range(sheet_name)
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Sheet {} not found", sheet_name)
                ))?
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

            let mut time_vec = Vec::new();
            let mut open_vec = Vec::new();
            let mut high_vec = Vec::new();
            let mut low_vec = Vec::new();
            let mut close_vec = Vec::new();
            let mut volume_vec = Vec::new();

            for row in range.rows().skip(1) {
                if row.len() >= 6 {
                    time_vec.push(row[0].to_string());
                    
                    open_vec.push(match &row[1] {
                        DataType::Float(f) => *f,
                        DataType::String(s) => s.parse::<f64>().unwrap_or(0.0),
                        _ => 0.0,
                    });

                    high_vec.push(match &row[2] {
                        DataType::Float(f) => *f,
                        DataType::String(s) => s.parse::<f64>().unwrap_or(0.0),
                        _ => 0.0,
                    });

                    low_vec.push(match &row[3] {
                        DataType::Float(f) => *f,
                        DataType::String(s) => s.parse::<f64>().unwrap_or(0.0),
                        _ => 0.0,
                    });

                    close_vec.push(match &row[4] {
                        DataType::Float(f) => *f,
                        DataType::String(s) => s.parse::<f64>().unwrap_or(0.0),
                        _ => 0.0,
                    });

                    volume_vec.push(match &row[5] {
                        DataType::Float(f) => *f,
                        DataType::String(s) => s.parse::<f64>().unwrap_or(0.0),
                        _ => 0.0,
                    });
                }
            }

            let df = DataFrame::new(vec![
                Series::new("time", time_vec),
                Series::new("open", open_vec),
                Series::new("high", high_vec),
                Series::new("low", low_vec),
                Series::new("close", close_vec),
                Series::new("volume", volume_vec),
            ]).map_err(convert_polars_err)?;

            self.data.insert(sheet_name.to_string(), df);
            Ok(())
        })
    }

    fn calculate_atr(&mut self, py: Python, sheet_name: &str, period: usize) -> PyResult<()> {
        py.allow_threads(|| {
            let df = self.data.get_mut(sheet_name)
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    format!("Sheet not found: {}", sheet_name)
                ))?;

            let high = series_to_vec(df.column("high").map_err(convert_polars_err)?)?;
            let low = series_to_vec(df.column("low").map_err(convert_polars_err)?)?;
            let close = series_to_vec(df.column("close").map_err(convert_polars_err)?)?;
            let n = high.len();

            let mut tr = vec![0.0; n];
            let mut atr = vec![0.0; n];

            // Calcular TR y ATR
            tr[0] = high[0] - low[0];
            for i in 1..n {
                tr[i] = (high[i] - low[i])
                    .max((high[i] - close[i-1]).abs())
                    .max((low[i] - close[i-1]).abs());
            }

            // Calcular ATR con suavizado exponencial
            let mut sum = tr[0..period].iter().sum::<f64>();
            atr[period-1] = sum / period as f64;

            for i in period..n {
                atr[i] = (atr[i-1] * (period - 1) as f64 + tr[i]) / period as f64;
            }

            df.with_column(Series::new("atr", atr))
                .map_err(convert_polars_err)?;

            Ok(())
        })
    }

    fn calculate_rsi(&mut self, py: Python, sheet_name: &str, period: usize) -> PyResult<()> {
        py.allow_threads(|| {
            let df = self.data.get_mut(sheet_name)
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    format!("Sheet not found: {}", sheet_name)
                ))?;

            let close = series_to_vec(df.column("close").map_err(convert_polars_err)?)?;
            let n = close.len();
            
            let mut gains = vec![0.0; n];
            let mut losses = vec![0.0; n];
            let mut rsi = vec![0.0; n];

            // Calcular ganancias y pérdidas
            for i in 1..n {
                let change = close[i] - close[i-1];
                if change > 0.0 {
                    gains[i] = change;
                } else {
                    losses[i] = -change;
                }
            }

            // Calcular promedio inicial de ganancias y pérdidas
            let mut avg_gain = gains[1..=period].iter().sum::<f64>() / period as f64;
            let mut avg_loss = losses[1..=period].iter().sum::<f64>() / period as f64;

            // Primera valor de RSI
            rsi[period] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss.max(f64::EPSILON)));

            // Calcular RSI para el resto de los períodos
            for i in (period + 1)..n {
                avg_gain = (avg_gain * (period as f64 - 1.0) + gains[i]) / period as f64;
                avg_loss = (avg_loss * (period as f64 - 1.0) + losses[i]) / period as f64;
                
                if avg_loss == 0.0 {
                    rsi[i] = 100.0;
                } else {
                    rsi[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss));
                }
            }

            // Redondear a 2 decimales
            let rsi: Vec<f64> = rsi.iter()
                .map(|x| (x * 100.0).round() / 100.0)
                .collect();

            df.with_column(Series::new("rsi", rsi))
                .map_err(convert_polars_err)?;

            Ok(())
        })
    }

    fn calculate_ema(&mut self, py: Python, sheet_name: &str, period: usize) -> PyResult<()> {
        py.allow_threads(|| {
            let df = self.data.get_mut(sheet_name)
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    format!("Sheet not found: {}", sheet_name)
                ))?;

            let close = series_to_vec(df.column("close").map_err(convert_polars_err)?)?;
            let n = close.len();
            let mut ema = vec![0.0; n];
            
            // Factor de suavizado
            let alpha = 2.0 / (period as f64 + 1.0);

            // Inicializar EMA con el primer valor
            ema[0] = close[0];

            // Calcular EMA
            for i in 1..n {
                ema[i] = close[i] * alpha + ema[i-1] * (1.0 - alpha);
            }

            // Redondear a 1 decimal
            let ema: Vec<f64> = ema.iter()
                .map(|x| (x * 10.0).round() / 10.0)
                .collect();

            df.with_column(Series::new(&format!("ema{}", period), ema))
                .map_err(convert_polars_err)?;

            Ok(())
        })
    }

    fn calculate_adx(&mut self, py: Python, sheet_name: &str, period: usize) -> PyResult<()> {
        py.allow_threads(|| {
            let df = self.data.get_mut(sheet_name)
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    format!("Sheet not found: {}", sheet_name)
                ))?;
    
            let high = series_to_vec(df.column("high").map_err(convert_polars_err)?)?;
            let low = series_to_vec(df.column("low").map_err(convert_polars_err)?)?;
            let close = series_to_vec(df.column("close").map_err(convert_polars_err)?)?;
            let n = high.len();
    
            let mut tr = vec![0.0; n];
            let mut plus_dm = vec![0.0; n];
            let mut minus_dm = vec![0.0; n];
            let mut smooth_plus_dm = vec![0.0; n];
            let mut smooth_minus_dm = vec![0.0; n];
            let mut smooth_tr = vec![0.0; n];
            let mut plus_di = vec![0.0; n];
            let mut minus_di = vec![0.0; n];
            let mut dx = vec![0.0; n];
            let mut adx = vec![0.0; n];
    
            // Calculate True Range and Directional Movement
            tr[0] = high[0] - low[0];
            for i in 1..n {
                // True Range
                tr[i] = (high[i] - low[i])
                    .max((high[i] - close[i-1]).abs())
                    .max((low[i] - close[i-1]).abs());
    
                // Directional Movement
                let up_move = high[i] - high[i-1];
                let down_move = low[i-1] - low[i];
    
                if up_move > down_move && up_move > 0.0 {
                    plus_dm[i] = up_move;
                }
                
                if down_move > up_move && down_move > 0.0 {
                    minus_dm[i] = down_move;
                }
            }
    
            // Calculate smoothed values using Wilder's smoothing (RMA)
            // First value is simple average
            for i in 0..period {
                smooth_tr[period-1] += tr[i];
                smooth_plus_dm[period-1] += plus_dm[i];
                smooth_minus_dm[period-1] += minus_dm[i];
            }
            
            smooth_tr[period-1] /= period as f64;
            smooth_plus_dm[period-1] /= period as f64;
            smooth_minus_dm[period-1] /= period as f64;
    
            // Calculate subsequent values using Wilder's smoothing
            for i in period..n {
                smooth_tr[i] = smooth_tr[i-1] * (period as f64 - 1.0) / period as f64 + tr[i] / period as f64;
                smooth_plus_dm[i] = smooth_plus_dm[i-1] * (period as f64 - 1.0) / period as f64 + plus_dm[i] / period as f64;
                smooth_minus_dm[i] = smooth_minus_dm[i-1] * (period as f64 - 1.0) / period as f64 + minus_dm[i] / period as f64;
                
                if smooth_tr[i] > 0.0 {
                    plus_di[i] = 100.0 * smooth_plus_dm[i] / smooth_tr[i];
                    minus_di[i] = 100.0 * smooth_minus_dm[i] / smooth_tr[i];
                }
            }
    
            // Calculate DX
            for i in period..n {
                if plus_di[i] + minus_di[i] > 0.0 {
                    dx[i] = 100.0 * (plus_di[i] - minus_di[i]).abs() / (plus_di[i] + minus_di[i]);
                }
            }
    
            // Calculate ADX using SMA of DX
            for i in (period*2)..n {
                let mut sum = 0.0;
                for j in (i-period+1)..=i {
                    sum += dx[j];
                }
                adx[i] = sum / period as f64;
            }
    
            // Update DataFrame
            df.with_column(Series::new("adx", adx))
                .map_err(convert_polars_err)?;
            df.with_column(Series::new("DIPlus", plus_di))
                .map_err(convert_polars_err)?;
            df.with_column(Series::new("DIMinus", minus_di))
                .map_err(convert_polars_err)?;
    
            Ok(())
        })
    }

    fn calculate_coral_trend(
        &mut self,
        py: Python,
        sheet_name: &str,
        smoothing_period: usize,
        constant_d: f64,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            let df = self.data.get_mut(sheet_name)
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    format!("Sheet not found: {}", sheet_name)
                ))?;

            let close = series_to_vec(df.column("close").map_err(convert_polars_err)?)?;
            let n = close.len();
            
            let di = (smoothing_period as f64 - 1.0) / 2.0 + 1.0;
            let c1 = 2.0 / (di + 1.0);
            let c2 = 1.0 - c1;
            let c3 = 3.0 * (constant_d * constant_d + constant_d * constant_d * constant_d);
            let c4 = -3.0 * (2.0 * constant_d * constant_d + constant_d + constant_d * constant_d * constant_d);
            let c5 = 3.0 * constant_d + 1.0 + constant_d * constant_d * constant_d + 3.0 * constant_d * constant_d;

            let mut i1 = vec![0.0; n];
            let mut i2 = vec![0.0; n];
            let mut i3 = vec![0.0; n];
            let mut i4 = vec![0.0; n];
            let mut i5 = vec![0.0; n];
            let mut i6 = vec![0.0; n];
            let mut bfr = vec![0.0; n];
            let mut colors = vec![String::from("gray"); n];

            i1[0] = close[0];
            i2[0] = close[0];
            i3[0] = close[0];
            i4[0] = close[0];
            i5[0] = close[0];
            i6[0] = close[0];

            for i in 1..n {
                i1[i] = c1 * close[i] + c2 * i1[i - 1];
                i2[i] = c1 * i1[i] + c2 * i2[i - 1];
                i3[i] = c1 * i2[i] + c2 * i3[i - 1];
                i4[i] = c1 * i3[i] + c2 * i4[i - 1];
                i5[i] = c1 * i4[i] + c2 * i5[i - 1];
                i6[i] = c1 * i5[i] + c2 * i6[i - 1];

                bfr[i] = -constant_d * constant_d * constant_d * i6[i] +
                         c3 * i5[i] + c4 * i4[i] + c5 * i3[i];

                colors[i] = if bfr[i] > bfr[i-1] {
                    String::from("green")
                } else if bfr[i] < bfr[i-1] {
                    String::from("red")
                } else {
                    String::from("blue")
                };
            }

            df.with_column(Series::new("Trend", bfr))
                .map_err(convert_polars_err)?;
            df.with_column(Series::new("Color", colors))
                .map_err(convert_polars_err)?;

            Ok(())
        })
    }

    fn apply_all_indicators(
        &mut self,
        py: Python,
        sheet_name: &str,
        smoothing_period: usize,
        adx_period: usize,
        rsi_period: usize,
    ) -> PyResult<()> {
        self.calculate_adx(py, sheet_name, adx_period)?;
        self.calculate_rsi(py, sheet_name, rsi_period)?;
        self.calculate_atr(py, sheet_name, 14)?;
        self.calculate_ema(py, sheet_name, 100)?;
        self.calculate_ema(py, sheet_name, 9)?;
        self.calculate_ema(py, sheet_name, 20)?;
        self.calculate_coral_trend(py, sheet_name, smoothing_period, 0.4)?;
        
        Ok(())
    }

    fn get_data_as_str(&self, sheet_name: &str) -> PyResult<String> {
        match self.data.get(sheet_name) {
            Some(df) => {
                let mut buf = Vec::new();
                
                // Configurar CsvWriter
                CsvWriter::new(&mut buf)
                .has_header(true)
                .finish(&mut df.clone()) // Usa una referencia mutable de un clon para evitar modificar el original
                .map_err(convert_polars_err)?;
                
                // Convertir el buffer a String
                let csv_str = String::from_utf8(buf)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                
                // Filtrar líneas de metadatos
                let filtered_csv: String = csv_str
                    .lines()
                    .filter(|line| !line.contains("shape:") && !line.trim().is_empty())
                    .collect::<Vec<&str>>()
                    .join("\n");
                
                Ok(filtered_csv)
            },
            None => Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("No data found for sheet {}", sheet_name)
            )),
        }
    }
}

#[pymodule]
fn market_data_processor(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<MarketData>()?;
    Ok(())
}