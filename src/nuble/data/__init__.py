"""NUBLE Data — Universe data, SEC EDGAR, FRED macro, S3 data lake pipelines."""

try:
    from .data_service import DataService, get_data_service
except ImportError:
    DataService = None
    get_data_service = None

try:
    from .s3_data_manager import S3DataManager, get_data_manager
except ImportError:
    S3DataManager = None
    get_data_manager = None

try:
    from .polygon_universe import PolygonUniverseData
except ImportError:
    PolygonUniverseData = None

try:
    from .sec_edgar import SECEdgarXBRL
except ImportError:
    SECEdgarXBRL = None

try:
    from .fred_macro import FREDMacroData
except ImportError:
    FREDMacroData = None

__all__ = [
    'DataService',
    'get_data_service',
    'S3DataManager',
    'get_data_manager',
    'PolygonUniverseData',
    'SECEdgarXBRL',
    'FREDMacroData',
]