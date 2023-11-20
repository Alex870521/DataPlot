def suggest_places(auth_key, city):
    """Returns longitude and latitude of first suggested location in the Netherlands from Postcode API.

    :param auth_key: authorization key for Postcode API
    :type auth_key: str
    :param city: textual input for city names to match in Postcode API
    :type city: str

    :rtype: (str, str), str, str
    :return: (longitude, latitude), Postcode API status code, Postcode API error message
    """

def add(num1, num2):
    """
    Add up two integer numbers.

    This function simply wraps the ``+`` operator, and does not
    do anything interesting, except for illustrating what
    the docstring of a very simple function looks like.

    Parameters
    ----------
    num1 : int
        First number to add.
    num2 : int
        Second number to add.

    Returns
    -------
    int
        The sum of ``num1`` and ``num2``.

    See Also
    --------
    subtract : Subtract one integer from another.

    Examples
    --------
    >>> add(2, 2)
    4
    >>> add(25, 0)
    25
    >>> add(10, -10)
    0
    """
    return num1 + num2


def plot(self, kind, color='blue', **kwargs):
    """
    Generate a config.

    Render the data in the Series as a matplotlib config of the
    specified kind.

    Parameters
    ----------
    kind : str
        Kind of matplotlib config.
    color : str
        Color name or rgb code.
    **kwargs
        These parameters will be passed to the matplotlib plotting
        function.
    """