import logging
import os
import time
import warnings

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Column, QTable
from astropy.wcs import WCS, FITSFixedWarning, NoWcsKeywordsFoundError
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from photutils.detection import DAOStarFinder

logging.getLogger("astroquery").setLevel(logging.WARNING)


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


def check_catalogs(ra_deg, dec_deg, search_radius=5 * u.arcsec, delay=1):
    """Проверка наличия объектов в каталогах Gaia DR3 и VSX.

    Args:
        ra_deg, dec_deg (float): Координаты в градусах (ICRS).
        search_radius: Радиус поиска (по умолчанию 5 угл. секунд).
        delay (int): Задержка между запросами в секундах (для ограничения нагрузки). По умолчанию 1.

    Returns:
        results (dict): Словарь с ключом - имя каталога, значение - имя найденного объекта.
        ** При модификации кода можно получить следующие характеристики объекта по запросу из каталога VSX: https://vizier.cds.unistra.fr/viz-bin/VizieR-3,
           а также из каталога Gaia DR3 https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html
    """

    results = {
        "Gaia DR3": ("Not checked"),
        "VSX": ("Not checked"),
    }

    coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit=(u.deg, u.deg), frame="icrs")

    try:
        # Проверка Gaia DR3
        time.sleep(delay)
        gaia_job = Gaia.cone_search_async(coordinate=coord, radius=search_radius)
        gaia_result = gaia_job.get_results()

        if gaia_result and "source_id" in gaia_result.colnames:
            results["Gaia DR3"] = str(gaia_result["source_id"][0])[:20]
        else:
            results["Gaia DR3"] = "Not found"
    except Exception:
        results["Gaia DR3"] = "Ошибка"

    try:
        # Проверка VSX
        time.sleep(delay)
        vsx_result = Vizier.query_region(
            coord, radius=search_radius, catalog="B/vsx/vsx"
        )

        if vsx_result and "Name" in vsx_result[0].colnames:
            results["VSX"] = vsx_result[0]["Name"][0].strip()[:30]
        else:
            results["VSX"] = "Not found"
    except Exception:
        results["VSX"] = "Ошибка"

    return results


def find_stars(data, fwhm=3.0, threshold=5.0):
    """Находит звёзды на снимке.

    Args:
        data (array or astropy.io.fits.hdu.base.DELAYED): Данные в HDU.
        fwhm (float): Полуширина главной оси ядра Гаусса в единицах пикселе (по умолчанию 3.0).
        threshold (float): Пороговое значение (по умолчанию 5.0).

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
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold * std)
    sources = daofind(data - median)

    if len(sources) == 0:
        raise ValueError("\nЗвезды не обнаружены на данном снимке.")
    else:
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


def check_catalogs_add2table(sources, search_radius=5 * u.arcsec, request_delay=1):
    """Проверяет наличие объектов в каждом каталоге и добавляет результат в таблицу.

    Args:
        sources (QTable): Таблица с международной системой координат.
        search_radius: Радиус поиска (по умолчанию 5 угл. секунд).
        request_delay (int): Задержка между запросами в секундах (для ограничения нагрузки). По умолчанию 1.

    Returns:
        sources (QTable): Таблица с добавлением новых колонок Gaia DR3 и VSX, в которых отображен результат поиска по каталогу.
    """

    # Добавление колонок для результатов
    for colname, length in [("Gaia DR3", 20), ("VSX", 30)]:
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
                ra, dec, search_radius=search_radius, delay=request_delay
            )
            for catalog, value in result.items():
                sources[catalog][idx] = value[: sources[catalog].dtype.itemsize // 4]
        except Exception:
            continue

    return sources


def results_to_csv(sources, fits_file):
    """Сохраняет результаты в csv файл."""

    output_file = f"results_{fits_file.split('.')[0]}.csv"
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
            if user_input:  # Не выводим ошибку при пустом вводе (если нет default)
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


def display_total_results(table):
    """Демонстрирует в виде таблицы результаты поиска звёзд в каталогах."""

    report = QTable()
    report["Статистика"] = [
        "Всего найдено звёзд,",
        "из которых:",
        "Найдено в каталоге Gaia",
        "Найдено в каталоге VSX",
        "Не найдено ни в одном каталоге",
        "Ошибок при выполнении",
    ]
    report["Количество"] = [
        len(table),
        "",
        (
            (table["Gaia DR3"] != "Not found")
            & (table["Gaia DR3"] != "Ошибка")
        ).sum(),
        (
            (table["VSX"] != "Not found") & (table["VSX"] != "Ошибка")
        ).sum(),
        ((table["Gaia DR3"] == "Not found") & (table["VSX"] == "Not found")).sum(),
        (
            (table["Gaia DR3"] == "Ошибка")
            | (table["VSX"] == "Ошибка")
        ).sum(),
    ]
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
            }

            catalog_params = {
                "search_radius": get_quantity(
                    "\nЗадайте радиус вокруг координат для поиска звёзд в каталогах через пробел, допустимые единицы измерения - arcsec, arcmin, deg. Например, 5.0 arcsec, 2.0 deg, ...",
                    "5.0 arcsec",
                ),
                "request_delay": get_number(
                    "\nВведите время задержки между запросами в секундах (для ограничения нагрузки),",
                    1,
                ),
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
            display_total_results(sources)

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


if __name__ == "__main__":
    main()
    input("\nНажмите Enter для выхода...")
