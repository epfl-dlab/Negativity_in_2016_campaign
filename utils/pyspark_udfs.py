import calendar
from dateutil.parser import parse as parseDate
import pyspark
from pyspark.sql import DataFrame
import pyspark.sql.functions as f
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, BooleanType, IntegerType, FloatType, StringType
from tld import get_tld
from typing import List, Tuple, Union


@udf(FloatType())
def arrayMax(arr: ArrayType(FloatType())) -> Union[float, int]:
    return float(max(arr.values))


@udf(IntegerType())
def arraySize(arr: ArrayType(FloatType())) -> int:
    return len(arr)


@udf(BooleanType())
def isDate(string: str) -> bool:
    """
    Returns true if the input string can be parsed to a date. Works for filtering non-date quotations.
    With help from: https://stackoverflow.com/questions/25341945/check-if-string-has-date-any-format
    Parameters
    ----------
    string: Arbitrary input string
    Returns
    -------
    Boolean indicator for "is string a date?".
    """
    try:
        parseDate(string, fuzzy=False, ignoretz=True)
        return True
    except Exception as exe:
        if not isinstance(exe, ValueError):
            if isinstance(exe, TypeError):
                pass  # Internal error of dateparser when handling the Value Error. So a type error is most likely actually a value error
            else:
                print('Caught unexpected Error', exe)
                print(exe)
        return False


def __URL2domain(url: str) -> str:
    """Trims the URL and leaves only the root domain."""
    ret = get_tld(url, as_object=True)
    return ret.domain + '.' + ret.tld


@udf(ArrayType(StringType()))
def URLs2domain(urls: List[str]) -> List[str]:
    """Trims a list of URLs and leaves only the root domains."""
    return [__URL2domain(url) for url in urls]
