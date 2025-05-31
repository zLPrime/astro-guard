import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_single_object(df, mag_col='mag', mjd_col='mjd', mag_err_col='mag_err'):
    """
    Полная предобработка и масштабирование данных одного объекта
    
    Параметры:
    df : pd.DataFrame
        Исходные данные с временными рядами
    mag_col : str, optional (default='mag')
        Название колонки с величиной звездной величины
    mjd_col : str, optional (default='mjd')
        Название колонки с Modified Julian Date
    mag_err_col : str, optional (default='mag_err')
        Название колонки с ошибкой звездной величины
    
    Возвращает:
    np.ndarray: подготовленные и масштабированные данные формы (1, 500, 3)
    или None при недостаточном количестве точек
    """
    # Приведение типов и проверка колонок
    required_cols = {mag_col, mjd_col, mag_err_col}
    for col in required_cols:
        if col not in df.columns:
            print(f"Отсутствует обязательная колонка: {col}")
            return None
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Создаем DataFrame с нужными колонками и стандартными именами
    processed_df = pd.DataFrame({
        'mag': df[mag_col],
        'mjd': df[mjd_col],
        'mag_err': df[mag_err_col]
    })
    
    # Базовая очистка данных
    processed_df.dropna(how='any', axis=0, inplace=True)
    processed_df.drop_duplicates(subset='mjd', inplace=True)
    
    # Фильтрация некорректных значений
    mask = (
        np.isfinite(processed_df['mag']) & 
        np.isfinite(processed_df['mjd']) & 
        np.isfinite(processed_df['mag_err'])
    )
    processed_df = processed_df[mask]
    
    # Фильтрация выбросов
    if not processed_df.empty:
        mean_err = processed_df['mag_err'].mean()
        mean_mag = processed_df['mag'].mean()
        std_mag = processed_df['mag'].std()
        
        err_mask = processed_df['mag_err'] < 3 * mean_err
        mag_mask = ((np.abs(processed_df['mag'] - mean_mag) / std_mag) < 5)
        processed_df = processed_df[err_mask & mag_mask]
    
    # Проверка минимального количества точек
    if len(processed_df) < 400:
        return None
    
    # Сортировка по времени
    processed_df.sort_values('mjd', inplace=True)
    values = processed_df[['mag', 'mag_err', 'mjd']].values
    
    # Применение паддинга
    padded = np.zeros((500, 3))
    valid_length = min(len(values), 500)
    padded[:valid_length] = values[:valid_length]
    
    # Масштабирование данных
    scaled_padded = padded.copy()
    non_zero_mask = padded[:, 2] != 0.0

    for ch in range(3):
        scaler = StandardScaler()
        valid_part = padded[non_zero_mask, ch].reshape(-1, 1)
        scaled_part = scaler.fit_transform(valid_part).flatten()
        scaled_padded[non_zero_mask, ch] = scaled_part
    
    return scaled_padded

def process_files(file_list, mag_col='mag', mjd_col='mjd', mag_err_col='mag_err'):
    """
    Обработка списка файлов с данными объектов
    
    Параметры:
    file_list : list of str
        Список путей к файлам с данными
        
    Возвращает:
    np.ndarray: массив данных всех объектов формы (n_objects, 500, 3)
    """
    X_ts = []
    
    for file in file_list:
        try:
            df = pd.read_csv(file)
            
            # Обработка одного объекта
            result = prepare_single_object(
                df, 
                mag_col=mag_col,
                mjd_col=mjd_col,
                mag_err_col=mag_err_col
            )
            if result is None:
                continue
                
            # Проверка формы результата
            if isinstance(result, np.ndarray) and result.shape == (500, 3):
                X_ts.append(result)
            else:
                print(f"Предупреждение: файл {file} вернул неожиданную форму {result.shape}")
                
        except Exception as e:
            print(f"Ошибка при обработке файла {file}: {str(e)}")
    
    return np.array(X_ts)