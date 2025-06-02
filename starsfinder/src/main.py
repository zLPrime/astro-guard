import logging
import os
import time
import tkinter as tk
import warnings
from tkinter import filedialog, messagebox, scrolledtext, ttk
from tkinter.font import Font
from tkinter.ttk import Progressbar, Style

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astroquery.exceptions import NoResultsWarning
from astropy.stats import sigma_clipped_stats
from astropy import conf
conf.max_lines = -1 
conf.max_width = -1 
from astropy.table import Column, QTable
from astropy.wcs import WCS, FITSFixedWarning, NoWcsKeywordsFoundError
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astroquery.mast import Catalogs
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.detection import DAOStarFinder

logging.getLogger("astroquery").setLevel(logging.WARNING)

warnings.filterwarnings("ignore", category=NoResultsWarning)
warnings.filterwarnings("ignore", category=FITSFixedWarning)
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*truncated right side string.*"
)
warnings.filterwarnings("ignore", module="photutils.detection")


def check_wcs(header):
    """Проверяет наличие корректного WCS в заголовке FITS.

    Args:
        header (Header): Заголовок HDU с метаданными.

    Raises:
        NoWcsKeywordsFoundError:
            Отсутствуют базовые ключи WCS,
            Отсутствуют параметры преобразования (CDELT или CD-матрица),
            CD-матрица содержит только нулевые значения",
            Некорректные значения в CD-матрице,
            Ошибка создания WCS. Используйте astrometry.net для калибровки изображения!
    """

    required_base = ["CTYPE1", "CRVAL1", "CRPIX1", "CTYPE2", "CRVAL2", "CRPIX2"]

    # Проверка базовых обязательных параметров
    missing_base = [k for k in required_base if k not in header]
    if missing_base:
        raise NoWcsKeywordsFoundError(
            f"\nОтсутствуют базовые ключи WCS: {missing_base}"
        )

    # Проверка наличия либо CDELT, либо CD-матрицы
    has_cdelt = all(k in header for k in ["CDELT1", "CDELT2"])
    has_cd = all(k in header for k in ["CD1_1", "CD1_2", "CD2_1", "CD2_2"])

    if not (has_cdelt or has_cd):
        raise NoWcsKeywordsFoundError(
            "\nОтсутствуют параметры преобразования (CDELT или CD-матрица)"
        )

    # Дополнительная проверка для CD-матрицы
    if has_cd:
        try:
            cd_matrix = [
                [header["CD1_1"], header["CD1_2"]],
                [header["CD2_1"], header["CD2_2"]],
            ]
            if all(v == 0 for row in cd_matrix for v in row):
                raise NoWcsKeywordsFoundError(
                    "\nCD-матрица содержит только нулевые значения"
                )
        except TypeError:
            raise NoWcsKeywordsFoundError("\nНекорректные значения в CD-матрице")

    # Попытка создания WCS для финальной проверки
    try:
        WCS(header)
    except Exception as e:
        raise NoWcsKeywordsFoundError(
            f"\nОшибка создания WCS: {str(e)}. Используйте astrometry.net для калибровки изображения!"
        )


def check_catalogs(ra_deg, dec_deg, search_radius=5 * u.arcsec, catalogs=None):
    """Проверка наличия объектов в каталогах Gaia DR3 и VSX.

    Args:
        ra_deg, dec_deg (float): Координаты в градусах (ICRS).
        search_radius: Радиус поиска (по умолчанию 5 угл. секунд).
        catalogs (list): Список каталогов для проверки. Если None, проверяются все.

    Returns:
        results (dict): Словарь с ключом - имя каталога, значение - имя найденного объекта.
        ** При модификации кода можно получить следующие характеристики объекта по запросу из каталога VSX: https://vizier.cds.unistra.fr/viz-bin/VizieR-3,
           а также из каталога Gaia DR3 https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html
    """

    # Все доступные каталоги
    all_catalogs = {
        "Gaia DR3": ("Not checked"),
        "VSX": ("Not checked"),
        "SIMBAD": ("Not checked"),
        "Pan-STARRS": ("Not checked"),
        "Hipparcos": ("Not checked")
    }
    
    # Если не указаны конкретные каталоги, используем все
    if catalogs is None:
        catalogs = list(all_catalogs.keys())
    
    # Инициализация результатов только для выбранных каталогов
    results = {cat: all_catalogs[cat] for cat in catalogs}

    coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit=(u.deg, u.deg), frame='fk5')
        
    # Проверка Gaia DR3
    if "Gaia DR3" in catalogs:
        try:
            time.sleep(1)
            gaia_job = Gaia.cone_search_async(coordinate=coord, radius=search_radius)
            gaia_result = gaia_job.get_results()

            if gaia_result and "source_id" in gaia_result.colnames:
                if len(gaia_result["source_id"]) > 0:
                    source_id = gaia_result["source_id"][0].astype(str)
                    results["Gaia DR3"] = source_id[:20]
                else:
                    results["Gaia DR3"] = "Not found"
            else:
                results["Gaia DR3"] = "Not found"
        except Exception:
            results["Gaia DR3"] = "Ошибка"

    # Проверка VSX
    if "VSX" in catalogs:
        try:
            time.sleep(1)
            vsx_result = Vizier.query_region(
                coord, radius=search_radius, catalog="B/vsx/vsx"
            )

            if vsx_result and "Name" in vsx_result[0].colnames:
                results["VSX"] = vsx_result[0]["Name"][0].strip()[:30]
            else:
                results["VSX"] = "Not found"
        except Exception:
            results["VSX"] = "Ошибка"
        
        
    # Поиск в SIMBAD
    if "SIMBAD" in catalogs:
        try:
            time.sleep(1)
            simbad_result = Simbad.query_region(coord, radius=search_radius)
            if simbad_result and len(simbad_result) > 0:
                results["SIMBAD"] = simbad_result["main_id"][0].strip()[:30]
            else:
                results["SIMBAD"] = "Not found"
        except Exception:
            results["SIMBAD"] = "Ошибка"
        
        
    # Поиск в Pan-STARRS
    if "Pan-STARRS" in catalogs:
        try:
            time.sleep(1)
            panstarrs_result = Catalogs.query_region(
                coord, 
                radius=search_radius,
                catalog="Panstarrs",
                data_release="dr2"
            )
            if panstarrs_result and len(panstarrs_result) > 0:
                results["Pan-STARRS"] = str(panstarrs_result["objID"][0])[:20]
            else:
                results["Pan-STARRS"] = "Not found"
        except Exception:
            results["Pan-STARRS"] = "Ошибка"
        
    # Hipparcos (I/239/hip_main)
    if "Hipparcos" in catalogs:
        try:
            time.sleep(1)
            hip_result = Vizier.query_region(
                coord, 
                radius=search_radius, 
                catalog="I/239/hip_main",
                cache=False
            )
            if hip_result and "HIP" in hip_result[0].colnames:
                results["Hipparcos"] = str(hip_result[0]["HIP"][0])[:20]
            else:
                results["Hipparcos"] = "Not found"
        except Exception:
            results["Hipparcos"] = "Ошибка"

    return results


def find_stars(data, fwhm=3.0, threshold=5.0, roundlo=-0.5):
    """Находит звёзды на снимке.

    Args:
        data (array or astropy.io.fits.hdu.base.DELAYED): Данные в HDU.
        fwhm (float): Полуширина главной оси ядра Гаусса в единицах пикселе (по умолчанию 3.0).
        threshold (float): Пороговое значение (по умолчанию 5.0).
        roundlo (float): Нижний порог круглости звезды (по умолчанию -0.5).

    Raises:
        ValueError: Звезды не обнаружены на данном снимке.

    Returns:
        sources (QTable or None): Таблица с обнаруженными звездами и их характеристиками.
            The table contains the following parameters:

            * ``id``: unique object identification number.
            * ``xcentroid, ycentroid``: object centroid.
            * ``sharpness``: object sharpness.
            * ``roundness1``: object roundness based on symmetry.
            * ``roundness2``: object roundness based on marginal Gaussian
              fits.
            * ``npix``: the total number of pixels in the Gaussian kernel
              array.
            * ``peak``: the peak pixel value of the object.
            * ``flux``: the object instrumental flux calculated as the
              sum of data values within the kernel footprint.
            * ``mag``: the object instrumental magnitude calculated as
              ``-2.5 * log10(flux)``.
            * ``daofind_mag``: the "mag" parameter returned by the DAOFIND
              algorithm. It is a measure of the intensity ratio of the
              amplitude of the best fitting Gaussian function at the
              object position to the detection threshold. This parameter
              is reported only for comparison to the IRAF DAOFIND
              output. It should not be interpreted as a magnitude
              derived from an integrated flux.
    """

    # найдём среднее, медиану и стандартное отклонение
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)

    # Поиск источников
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold * std, roundlo=roundlo)
    sources = daofind(data - median)

    if len(sources) == 0:
        raise ValueError("\nЗвезды не обнаружены на данном снимке.")
    else:
        # Апертурная фотометрия для расчета flux_error
        radius = 2.0 * fwhm # Радиус апертуры = 2 * FWHM
        positions = list(zip(sources["xcentroid"], sources["ycentroid"]))
        apertures = CircularAperture(positions, r=radius)
        
        # Измеряем поток в апертуре (с вычетом фона)
        phot_table = aperture_photometry(data - median, apertures)
        flux = phot_table["aperture_sum"].data  # Поток из апертурной фотометрии
        flux_error = np.sqrt(apertures.area) * std

        # Расчет mag_error (исправленная версия)
        mag_error = (2.5 / np.log(10)) * (flux_error / flux)
        
        # Добавляем колонки в таблицу
        sources.add_column(Column(flux, name="aperture_flux"))
        sources.add_column(Column(flux_error, name="flux_err"))
        sources.add_column(Column(mag_error, name="mag_err", description="Ошибка магнитуды (из aperture_flux)"))

    
        return sources


def pixel_to_wcs(sources, header):
    """Преобразует координаты звёзд из пиксельных в международную систему координат WCS и добавляет новые значения в таблицу.

    Args:
        sources (QTable): Таблица со звездами и их характеристиками.
        header (Header): Заголовок HDU с метаданными.

    Returns:
        sources (QTable): Таблица с добавлением международной системы координат.
    """

    wcs = WCS(header)
    coords = wcs.pixel_to_world(sources["xcentroid"], sources["ycentroid"])
    sources.add_column(Column(coords.ra.deg, name="ra", dtype="float64"))
    sources.add_column(Column(coords.dec.deg, name="dec", dtype="float64"))

    return sources


def check_catalogs_add2table(sources, search_radius=5 * u.arcsec, catalogs=None):
    """Проверяет наличие объектов в каждом каталоге и добавляет результат в таблицу.

    Args:
        sources (QTable): Таблица с международной системой координат.
        search_radius: Радиус поиска (по умолчанию 5 угл. секунд).
        catalogs (list): Список каталогов для проверки. Если None, проверяются все.

    Returns:
        sources (QTable): Таблица с добавлением колонок для выбранных каталогов.
    """

    # Все доступные каталоги
    all_catalogs = ["Gaia DR3", "VSX", "SIMBAD", "Pan-STARRS", "Hipparcos"]
    
    # Если не указаны конкретные каталоги, используем все
    if catalogs is None:
        catalogs = all_catalogs
    
    # Добавление колонок только для выбранных каталогов
    for colname in catalogs:
        length = 30  # Максимальная длина строки
        sources.add_column(
            Column(
                data=np.full(len(sources), "Not found", dtype=f"U{length}"),
                name=colname,
            )
        )

    # Проверка каждого источника
    for idx in range(len(sources)):
        ra = sources["ra"][idx]
        dec = sources["dec"][idx]

        try:
            result = check_catalogs(
                ra, dec, 
                search_radius=search_radius,
                catalogs=catalogs
            )
            for catalog, value in result.items():
                sources[catalog][idx] = value[: sources[catalog].dtype.itemsize // 4]
        except Exception:
            continue

    return sources


def results_to_csv(sources, fits_file):
    """Сохраняет результаты в csv файл."""

    output_file = f"results_{fits_file.replace('\\', '.').split('.')[-2]}.csv"
    sources.write(output_file, overwrite=True, format="csv")
    print(f"\nРезультаты сохранены в {output_file}")


def get_number(prompt, default=None):
    """Просит пользователя ввести простое числовое значение параметра для функции."""

    # Добавляем подсказку о значении по умолчанию
    if default is not None:
        prompt += f" [по умолчанию: {default}].\nЧтобы оставить значение по-умолчанию, нажмите Enter. Иначе введите своё число: "

    while True:
        user_input = input(prompt).strip()

        # Если ввод пустой и есть значение по умолчанию
        if not user_input and default is not None:
            return default

        # Если ввод не пустой, пытаемся преобразовать
        try:
            return float(user_input)
        except ValueError:
            if user_input: 
                print("\nОшибка! Введите число")
            else:
                print("\nОшибка! Значение не может быть пустым")


def get_quantity(prompt, default=None):
    """Запрашивает у пользователя значение с единицами измерения."""

    # Единицы измерения, допустимые для ввода
    units_list = ['arcmin', 'arcsec', 'deg']
    
    if default is not None:
        prompt += f" [по умолчанию: {default}].\nЧтобы оставить значение по-умолчанию, нажмите Enter. Иначе введите своё число: "

    while True:
        user_input = input(prompt).strip()

        # Если ввод пустой и есть значение по умолчанию
        if not user_input and default is not None:
            return default

        if not user_input:
            print("\nОшибка! Введите значение")
            continue

        # Если ввод не пустой, пытаемся преобразовать
        parts = user_input.split()
        try:
            value = float(parts[0])
            if len(parts) == 1:
                print('Вы забыли ввести единицу измерения!')
                continue
            else:
                unit_str = parts[1]
                if unit_str not in units_list:
                    print('Недопустимые единицы измерения!')
                    continue
                else:
                    unit = u.Unit(unit_str)
            return value * unit
        
        except (ValueError, TypeError):
            print("\nОшибка формата. Пример: '3.0 arcsec'")
            

def get_catalog_selection():
    """Запрашивает у пользователя выбор каталогов для проверки."""
    
    available_catalogs = {
        "1": "Gaia DR3",
        "2": "VSX",
        "3": "SIMBAD",
        "4": "Pan-STARRS",
        "5": "Hipparcos"
    }
    
    print("\nДоступные каталоги для проверки:")
    for num, cat in available_catalogs.items():
        print(f"{num}. {cat}")
    
    print("\nВведите номера каталогов через запятую (например '1,2,3')")
    print("Или нажмите Enter для проверки всех каталогов")
    
    while True:
        user_input = input("Ваш выбор: ").strip()
        
        if not user_input:
            return None  # Проверять все каталоги
        
        try:
            selected = []
            for num in user_input.split(','):
                num = num.strip()
                if num in available_catalogs:
                    selected.append(available_catalogs[num])
                else:
                    raise ValueError(f"Неизвестный номер каталога: {num}")
            
            if not selected:
                raise ValueError("Не выбрано ни одного каталога")
                
            return selected
            
        except ValueError as e:
            print(f"Ошибка: {str(e)}. Попробуйте снова.")
            

def display_total_results(table, catalogs=None):
    """Демонстрирует в виде таблицы результаты поиска звёзд в каталогах.
    
    Args:
        table (QTable): Таблица с результатами поиска
        catalogs (list): Список выбранных каталогов. Если None, учитываются все.
    """
    
    # Все доступные каталоги
    all_catalogs = ["Gaia DR3", "VSX", "SIMBAD", "Pan-STARRS", "Hipparcos"]
    
    # Если не указаны конкретные каталоги, используем все
    if catalogs is None:
        catalogs = all_catalogs
    
    # Проверяем, какие каталоги действительно есть в таблице
    available_in_table = [cat for cat in catalogs if cat in table.colnames]
    
    # Собираем данные в списки перед созданием таблицы
    stats = ["Всего найдено звёзд", ""]
    counts = [len(table), ""]
    
    # Добавляем статистику по каждому каталогу
    for catalog in available_in_table:
        found = ((table[catalog] != "Not found") & (table[catalog] != "Ошибка")).sum()
        stats.append(f"Найдено в {catalog}")
        counts.append(found)
    
    # Добавляем статистику по не найденным объектам
    not_found_condition = None
    for catalog in available_in_table:
        if not_found_condition is None:
            not_found_condition = (table[catalog] == "Not found")
        else:
            not_found_condition &= (table[catalog] == "Not found")
    
    if not_found_condition is not None:
        stats.append("Не найдено ни в одном каталоге")
        counts.append(not_found_condition.sum())
    
    # Добавляем статистику по ошибкам
    error_condition = None
    for catalog in available_in_table:
        if error_condition is None:
            error_condition = (table[catalog] == "Ошибка")
        else:
            error_condition |= (table[catalog] == "Ошибка")
    
    if error_condition is not None:
        stats.append("Ошибок при выполнении")
        counts.append(error_condition.sum())
    
    # Создаем таблицу из собранных данных
    report = QTable()
    report["Статистика"] = stats
    report["Количество"] = counts
    
    print(report)


def main():
    """Запускает и осуществляет пайплйн всего файла:
        - Запрашивает у пользователя файл fits
        - Осуществляет проверку файла
        - Находит звезды на снимке
        - Переводит пиксельные координаты звёзд в международные (wcs)
        - По найденным координатам ищет звездные объекты в международных каталогах
        - Выводит таблицу с результатами поиска и сохраняет результаты при необходимости

    Raises:
        ValueError:
            Выбранный HDU не содержит данных!,
            Некорректные данные (наличие пустых пикселей!,
            Некорректные данные (наличие отрицательных пикселей!
    """

    # Запрос пути к файлу и проверка его существования
    file_path = input("\nВведите путь к FITS файлу: ").strip()

    if not os.path.exists(file_path):
        print("\nОшибка: Файл не найден!")
        return

    try:
        # Открытие файла и выбор HDU
        with fits.open(file_path) as hdul:
            print("\nСтруктура файла: \n")
            hdul.info()

            # Выбор HDU
            while True:
                try:
                    hdu_num = int(input("\nВведите номер (№) HDU для анализа: "))
                    hdu = hdul[hdu_num]
                    break
                except (ValueError, IndexError):
                    print("\nНеверный ввод. Попробуйте снова.")

            header = hdu.header
            data = hdu.data

            # Проверка данных на пустые значения
            if (data is None) or data.size == 0:
                raise ValueError("\nВыбранный HDU не содержит данных!")

            if np.isnan(data).any():
                raise ValueError("\nНекорректные данные (наличие пустых пикселей!)")

            if data[data < 0].any():
                raise ValueError(
                    "\nНекорректные данные (наличие отрицательных пикселей!)"
                )

            # Проверка коррекного WCS
            check_wcs(header)

            # Конфигурация поиска звезд
            detection_params = {
                "fwhm": get_number(
                    "\nВведите полуширину главной оси ядра Гаусса в единицах пикселей (fwhm) для обнаружения звезд на снимке,",
                    3.0,
                ),
                "threshold": get_number(
                    "\nТеперь введите пороговое значение (threshold) для обнаружения звезд на снимке,",
                    5.0,
                ),
                "roundlo": get_number(
                    "\nВведите нижний порог круглости звезды (roundlo),",
                    -0.5,
                ),
            }

            catalog_params = {
                "search_radius": get_quantity(
                    "\nЗадайте радиус вокруг координат для поиска звёзд в каталогах через пробел, допустимые единицы измерения - arcsec, arcmin, deg. Например, 5.0 arcsec, 2.0 deg, ...",
                    "5.0 arcsec",
                ),
                "catalogs": get_catalog_selection()
            }

            print("\nОбработка началась. Ожидайте...")

            # Конвейер обработки данных
            sources = check_catalogs_add2table(
                sources=pixel_to_wcs(
                    sources=find_stars(data, **detection_params), header=header
                ),
                **catalog_params,
            )

            # Выведем полученные результаты
            print("\nПервые 5 строк полученной таблицы: \n", sources[0:5])
            print("\n")
            display_total_results(sources, catalog_params.get("catalogs"))

            # Запрос на сохранение таблицы с результатами в формате csv
            while True:
                user_input = input("\nСохранить таблицу? (1 - Да, 0 - Нет): ").strip()
                if user_input in ("1", "0"):
                    break
                print("\nОшибка: введите 1 или 0!")

            if user_input == "1":
                results_to_csv(sources, file_path)
            else:
                print("\nСохранение отменено")

    except Exception as e:
        print(f"\nОшибка при обработке файла: {str(e)}")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("📡 Анализатор астрономических каталогов")
        self.geometry("1100x850")
        self.current_hdul = None
        self.last_results = None
        
        # Настройка цветовой схемы
        self.colors = {
            'primary': '#2C3E50',
            'secondary': '#3498DB',
            'success': '#27AE60',
            'danger': '#E74C3C',
            'background': '#FFFFFF',
            'text': '#2C3E50',
            'result_bg': '#F8F9FA'
        }
        
        # Настройка шрифтов
        self.title_font = Font(family="Segoe UI", size=14, weight="bold")
        self.base_font = Font(family="Segoe UI", size=11)
        self.mono_font = Font(family="Consolas", size=10)
        
        # Инициализация стилей
        self.style = Style()
        self.style.theme_use('clam')
        self.configure_styles()
        
        self.create_widgets()
        self.set_defaults()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def configure_styles(self):
        """Настройка кастомных стилей для виджетов"""
        self.configure(bg=self.colors['background'])
        self.style = Style()
        self.style.theme_use('clam')
        self.style.configure(
            '.', 
            background=self.colors['background'],
            foreground=self.colors['text'],
            font=self.base_font
        )
        
        self.style.configure(
            'TButton',
            background=self.colors['secondary'],
            foreground='white',
            borderwidth=1,
            focusthickness=3,
            focuscolor=self.colors['secondary']
        )
        self.style.map('TButton',
            background=[('active', self.colors['primary'])]
        )
        
        self.style.configure(
            'Header.TLabel', 
            font=self.title_font,
            foreground=self.colors['primary'],
            background=self.colors['background']
        )
        
        self.style.configure(
            'TCombobox',
            selectbackground=self.colors['secondary']
        )

    def create_widgets(self):
        # Основной контейнер
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Секция выбора файла
        file_frame = ttk.LabelFrame(
            main_frame, 
            text=" 🗃️ Выбор FITS файла", 
            style='Header.TLabel'
        )
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.file_entry = ttk.Entry(file_frame, width=85)
        self.file_entry.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        ttk.Button(
            file_frame, 
            text="Обзор...", 
            style='TButton',
            command=self.browse_file
        ).pack(side=tk.LEFT, padx=5)

        # Секция выбора HDU
        hdu_frame = ttk.LabelFrame(
            main_frame, 
            text=" 📂 Выбор HDU", 
            style='Header.TLabel'
        )
        hdu_frame.pack(fill=tk.X, pady=10)
        
        self.hdu_selector = ttk.Combobox(
            hdu_frame, 
            state="readonly",
            font=self.base_font
        )
        self.hdu_selector.pack(padx=5, pady=5, fill=tk.X)

        # Параметры обработки
        params_frame = ttk.Frame(main_frame)
        params_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # Секция обнаружения
        detect_frame = ttk.LabelFrame(
            params_frame,
            text=" 🔭 Параметры обнаружения"
        )
        detect_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Сетка обнаружения
        ttk.Label(detect_frame, text="FWHM (пикс):").grid(row=0, column=0, padx=5, pady=2, sticky='e')
        self.fwhm_entry = ttk.Entry(detect_frame, width=10)
        self.fwhm_entry.grid(row=0, column=1, padx=5, pady=2, sticky='w')

        ttk.Label(detect_frame, text="Порог (σ):").grid(row=1, column=0, padx=5, pady=2, sticky='e')
        self.threshold_entry = ttk.Entry(detect_frame, width=10)
        self.threshold_entry.grid(row=1, column=1, padx=5, pady=2, sticky='w')

        ttk.Label(detect_frame, text="Круглость (min):").grid(row=2, column=0, padx=5, pady=2, sticky='e')
        self.roundlo_entry = ttk.Entry(detect_frame, width=10)
        self.roundlo_entry.grid(row=2, column=1, padx=5, pady=2, sticky='w')

        # Секция каталогов
        catalog_frame = ttk.LabelFrame(
            params_frame,
            text=" 📚 Параметры каталогов"
        )
        catalog_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # Фрейм для выбора каталогов
        catalog_select_frame = ttk.Frame(catalog_frame)
        catalog_select_frame.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 5))

        ttk.Label(catalog_select_frame, text="Каталоги:").pack(side=tk.LEFT, padx=5)
        
        # Создаем переменные для чекбоксов каталогов
        self.catalog_vars = {
            'Gaia DR3': tk.BooleanVar(value=True),
            'VSX': tk.BooleanVar(value=True),
            'SIMBAD': tk.BooleanVar(value=False),
            'Pan-STARRS': tk.BooleanVar(value=False),
            'Hipparcos': tk.BooleanVar(value=False)
        }
        
        # Создаем чекбоксы для каждого каталога
        for catalog in self.catalog_vars:
            cb = ttk.Checkbutton(
                catalog_select_frame,
                text=catalog,
                variable=self.catalog_vars[catalog],
                onvalue=True,
                offvalue=False
            )
            cb.pack(side=tk.LEFT, padx=2)

        # Сетка каталогов (радиус и единицы измерения)
        ttk.Label(catalog_frame, text="Радиус:").grid(row=1, column=0, padx=5, pady=2, sticky='e')
        self.radius_entry = ttk.Entry(catalog_frame, width=10)
        self.radius_entry.grid(row=1, column=1, padx=5, pady=2, sticky='w')

        self.radius_units = ttk.Combobox(
            catalog_frame,
            values=["arcsec", "arcmin", "deg"],
            width=8,
            state="readonly"
        )
        self.radius_units.grid(row=1, column=2, padx=5, pady=2, sticky='w')

        # Управление
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=15)
        
        self.run_btn = ttk.Button(
            control_frame, 
            text="🚀 Запустить обработку", 
            style='TButton',
            command=self.run_processing
        )
        self.run_btn.pack(side=tk.LEFT, padx=5, ipadx=10)
        
        self.save_btn = ttk.Button(
            control_frame, 
            text="💾 Сохранить CSV", 
            style='TButton',
            state=tk.DISABLED, 
            command=self.save_results
        )
        self.save_btn.pack(side=tk.LEFT, padx=5, ipadx=10)

        # Прогресс-бар
        self.progress = Progressbar(
            main_frame, 
            orient=tk.HORIZONTAL, 
            mode='indeterminate',
            style='TProgressbar'
        )
        self.progress.pack(fill=tk.X, pady=(10, 15))

        # Результаты
        result_frame = ttk.LabelFrame(
            main_frame, 
            text=" 📊 Результаты обработки", 
            style='Header.TLabel'
        )
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_text = scrolledtext.ScrolledText(
            result_frame, 
            wrap=tk.WORD,
            font=self.mono_font,
            bg=self.colors['result_bg'],
            padx=12,
            pady=12,
            tabs=('4cm', 'right'),
            insertbackground=self.colors['text']
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
    def set_defaults(self):
        self.fwhm_entry.insert(0, "3.0")
        self.threshold_entry.insert(0, "5.0")
        self.roundlo_entry.insert(0, "-0.5") 
        self.radius_entry.insert(0, "5.0")
        self.radius_units.current(0)

    def browse_file(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("FITS files", "*.fits"), ("All files", "*.*")]
        )
        if filepath:
            try:
                # Закрываем предыдущий файл
                if self.current_hdul:
                    self.current_hdul.close()
                    self.current_hdul = None
                
                # Открываем новый файл
                self.current_hdul = fits.open(filepath)
                
                # Обновляем список HDU
                hdu_list = [
                    f"{i}: {hdu.name} ({hdu.header.get('NAXIS', '?')}D)" 
                    for i, hdu in enumerate(self.current_hdul)
                ]
                self.hdu_selector.config(values=hdu_list)
                self.hdu_selector.current(0)
                
                self.file_entry.delete(0, tk.END)
                self.file_entry.insert(0, filepath)
                self.save_btn.config(state=tk.DISABLED)
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка открытия файла: {str(e)}")

    def validate_inputs(self):
        required = [
            (self.file_entry, "Выберите FITS файл"),
            (self.hdu_selector, "Выберите HDU"),
            (self.fwhm_entry, "Введите FWHM"),
            (self.threshold_entry, "Введите пороговое значение"),
            (self.radius_entry, "Введите радиус поиска")
        ]
        
        for field, msg in required:
            if isinstance(field, ttk.Entry) and not field.get().strip():
                messagebox.showerror("Ошибка", msg)
                return False
            elif isinstance(field, ttk.Combobox) and field.current() < 0:
                messagebox.showerror("Ошибка", msg)
                return False
        
        try:
            float(self.fwhm_entry.get())
            float(self.threshold_entry.get())
            float(self.radius_entry.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные числовые значения")
            return False
            
        return True

    def run_processing(self):
        if not self.validate_inputs():
            return
            
        try:
            self.run_btn.config(state=tk.DISABLED)
            self.save_btn.config(state=tk.DISABLED)
            self.progress.start()
            self.result_text.delete(1.0, tk.END)
            
            # Собираем параметры
            params = {
                'detection_params': {
                    'fwhm': float(self.fwhm_entry.get()),
                    'threshold': float(self.threshold_entry.get()),
                    'roundlo': float(self.roundlo_entry.get()),
                },
                'catalog_params': {
                    'search_radius': f"{self.radius_entry.get()} {self.radius_units.get()}"
                }
            }
            
            # Обработка данных
            self.last_results = self.process_file(params)  # Сохраняем результаты
            
            # Вывод результатов
            self.result_text.insert(tk.END, "Первые 5 строк таблицы:\n")
            self.result_text.insert(tk.END, str(self.last_results[0:5]) + "\n\n")
            self.result_text.insert(tk.END, "Статистика:\n")
            self.display_total_results(self.last_results)
            
            self.save_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
        finally:
            self.progress.stop()
            self.run_btn.config(state=tk.NORMAL)

    def process_file(self, params):
        try:
            if not self.current_hdul:
                raise RuntimeError("Файл не открыт")
            
            hdu_index = self.hdu_selector.current()
            hdu = self.current_hdul[hdu_index]
            
            # Проверки данных
            if hdu.data is None or hdu.data.size == 0:
                raise ValueError("Выбранный HDU не содержит данных!")
                
            if np.isnan(hdu.data).any():
                raise ValueError("Наличие NaN в данных!")
                
            if np.any(hdu.data < 0):
                raise ValueError("Наличие отрицательных значений в данных!")
            
            # Проверка WCS
            check_wcs(hdu.header)
            
            # Получаем список выбранных каталогов
            selected_catalogs = [cat for cat, var in self.catalog_vars.items() if var.get()]
            
            # Основная обработка
            sources = check_catalogs_add2table(
                pixel_to_wcs(
                    find_stars(hdu.data, **params['detection_params']),
                    hdu.header
                ),
                search_radius=u.Quantity(params['catalog_params']['search_radius']),
                catalogs=selected_catalogs
            )
            
            return sources
            
        except Exception as e:
            raise RuntimeError(f"Ошибка обработки: {str(e)}")

    def display_total_results(self, table):
        # Получаем список каталогов, которые есть в таблице
        available_catalogs = [col for col in table.colnames if col in self.catalog_vars]
        
        # Собираем статистику
        stats = []
        
        # Общее количество звезд
        stats.append(("🌟 Всего найдено звёзд:", len(table)))
        
        # Для каждого каталога добавляем статистику
        for catalog in available_catalogs:
            found = ((table[catalog] != "Not found") & (table[catalog] != "Ошибка")).sum()
            stats.append((f"🌌 Совпадений с {catalog}:", found))
        
        # Не идентифицированные объекты (не найдены ни в одном каталоге)
        not_found_condition = None
        for catalog in available_catalogs:
            if not_found_condition is None:
                not_found_condition = (table[catalog] == "Not found")
            else:
                not_found_condition &= (table[catalog] == "Not found")
        
        if not_found_condition is not None:
            stats.append(("🔍 Не идентифицировано:", not_found_condition.sum()))
        
        # Ошибки запросов
        error_condition = None
        for catalog in available_catalogs:
            if error_condition is None:
                error_condition = (table[catalog] == "Ошибка")
            else:
                error_condition |= (table[catalog] == "Ошибка")
        
        if error_condition is not None:
            stats.append(("⚠️ Ошибок запросов:", error_condition.sum()))
        
        # Выводим результаты
        self.result_text.configure(state='normal')
        self.result_text.delete(1.0, tk.END)
        
        # Заголовок
        self.result_text.tag_configure('header', font=self.title_font, foreground=self.colors['primary'])
        self.result_text.insert(tk.END, "Результаты обработки\n", 'header')
        
        # Данные
        self.result_text.tag_configure('data', lmargin1=20, lmargin2=40)
        self.result_text.tag_configure('num', foreground=self.colors['secondary'])
   
        for name, value in stats:
            # Форматируем строку с фиксированной шириной
            self.result_text.insert(tk.END, f"{name}\t", 'data')
            self.result_text.insert(tk.END, f"{value}\n", ('data', 'num'))
        
        # Настройки форматирования
        from astropy.table import conf
        conf.max_lines = None
        conf.max_width = -1
        conf.max_columns = -1
        
        # Выводим пример данных только для выбранных каталогов
        columns = ['id', 'xcentroid', 'ycentroid'] + available_catalogs
        output = table[columns]
        for col in ['xcentroid', 'ycentroid']:
            output[col].format = "{:.3f}"
            
        self.result_text.insert(tk.END, "\nПример данных (первые 5 строк):\n", 'header')
        self.result_text.insert(tk.END, str(output[:5]))
        self.result_text.configure(state='disabled')
        
        
    def save_results(self):
        if self.last_results is None:
            messagebox.showerror("Ошибка", "Нет данных для сохранения")
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.last_results.write(filepath, overwrite=True, format='csv')
                messagebox.showinfo("Успех", f"Файл успешно сохранен:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при сохранении файла:\n{str(e)}")

    def on_close(self):
        if self.current_hdul:
            self.current_hdul.close()
        if messagebox.askokcancel("Выход", "Вы уверены что хотите выйти?"):
            self.destroy()


if __name__ == "__main__":
    choice = input("Выберите режим (1 - GUI, 2 - Консоль): ")
    
    if choice == "1":
        app = App()
        app.mainloop()
    elif choice == "2":
        main()
    else:
        print("Некорректный выбор")