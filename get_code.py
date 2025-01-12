def get_prompt_for_code_generation(alpha_description, samples=None):
    return f""" Firstly, I will introduce basic information about the input data:
1. The input data, referred to as 'df', is a Python dictionary that contains six DataFrames obtained from baostock and provides historical data for a specific stock, named 'open', 'close', 'high', 'low', 'volume', 'vwap'
2. Each DataFrame's columns correspond to stock names, and the rows are indexed by a DateTimeIndex. 
3. Each of them represents a specific aspect of the stock's performance and financial metrics, enabling users to track its fluctuations over time.
4. This data will be loaded and used in the subsequent code you generate.

Features in `df`(true feature dtypes listed here): 
{samples}

The functions and operators used in the alphas are defined below: (All expressions are case insensitive.)
1. Same for the operators “+”, “-”, “*”, “/”, “>”, “<”, “==”, “||”, “x ? y : z”
2. Common functions and their explanations as follows, some of which include code example('yy' represents the alias of a certain feature):
    2.1 abs(x), log(x), sign(x) = standard definitions; 
    2.2 rank(x) = cross-sectional rank of stocks, it means all stocks are ranked at the same time point. Code example: df['yy'].rank(axis=1);
    2.3 delay(x, d) = value of x d days ago;
    2.4 correlation(x, y, d) = time-serial correlation of x and y for the past d days;
    2.5 covariance(x, y, d) = time-serial covariance of x and y for the past d days;
    2.6 scale(x, a) = rescaled x such that sum(abs(x)) = a (the default is a = 1). Code example: df['yy'].apply(lambda row: row * k / np.abs(row).sum(), axis=1);
    2.7 delta(x, d) = today’s value of x minus the value of x d days ago;
    2.8 signedpower(x, a) = x^a;
    2.9 decay_linear(x, d) = weighted moving average over the past d days with linearly decaying weights d, d – 1, ..., 1 (rescaled to sum up to 1);
    2.10 ts_O(x, d) = operator O applied across the time-series for the past d days; non-integer number of days d is converted to floor(d);
    2.11 ts_min(x, d) = time-series min over the past d day;
    2.12 ts_max(x, d) = time-series max over the past d days;
    2.13 ts_argmax(x, d) = which day ts_max(x, d) occurred on;
    2.14 ts_argmin(x, d) = which day ts_min(x, d) occurred on;
    2.15 ts_rank(x, d) = time-series rank in the past d days;
    2.16 sum(x, d) = time-series sum over the past d days;
    2.17 product(x, d) = time-series product over the past d days;
    2.18 stddev(x, d) = moving time-series standard deviation over the past d days.
    
Task:
    1. Given the `df` and the Alpha Description, you need to write a python code to implement the quantitative Alpha.
    2. This code should generate a new additional feature called `alpha` to `df` that serves as a key metric in evaluating the effectiveness and skill of investment strategies, helping investors identify opportunities for superior returns. Make sure all used features exist. Follow the above description of features closely and consider the datatypes and meanings of operators and functions used in Alpha. Make sure the generated code does not use future information.

There are some important notes that you should follow: 
    1. rank(x) is totally different from ts_rank(x,d). rank(x) is cross-sectional rank, while ts_rank(x, d) is time-series rank in the past d days. If you need to implement 'rank(x)', you should use 'x.rank(axis=1, pct=True)' or other codes to implement cross-sectional rank, you cannot use 'x.rank(pct=True)' to implement this function, because it's time-series rank.
    2. Please remember that the new feature you generate must be formatted as a DataFrame. Be sure to use pd.DataFrame(x) to align the format in a timely manner, and do not represent it as np.ndarray or any other format.
    3. You should remember that each codeblock ends with ```end and starts with "```python".

Code formatting for each codeblock:
```python
# (Alpha description)
# ... Finally you need to add a new feature named 'alpha' to `df`: df['alpha'] = ...
```end

Example:
Alpha Description:
(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) -0.5)

Codeblock:
```python
# Calculate daily returns
df['returns'] = df['close'].pct_change()
# Calculate standard deviation of returns over a 20-day period
df['stddev_returns_20'] = df['returns'].rolling(window=20).std()
# Create a mask to identify negative returns
negative_returns_mask = df['returns'] < 0
# Compute the SignedPower factor, using either the standard deviation or the closing price, based on the sign of returns
# remember to change 'np.ndarray' to pd.dataframe
df['SignedPower'] = pd.DataFrame(np.where(negative_returns_mask, df['stddev_returns_20'], df['close']) ** 2)
# Rank the Ts_ArgMax (index of the maximum value) over the past 5 days
df['Ts_ArgMax'] = df['SignedPower'].rolling(window=5).apply(lambda x: np.argmax(x), raw=True)
# Rank the results and subtract 0.5
df['alpha'] = df['Ts_ArgMax'].rank(axis=1, pct=True) - 0.5
```end

Then, I will give you an alpha description, please complete the codeblock
Alpha Description:
{alpha_description}
Codeblock:
"""


def build_prompt_from_df_for_code_generation(df, alpha_description):
    samples = ""
    features = ['open', 'close', 'high', 'low', 'volume', 'vwap']
    for feature in features:
        samples = samples + f"The samples for `df['{feature}']` is:\n"
        df_ = df[feature].iloc[:2, :5]
        samples += df_.to_string()
        samples += '\n'

    prompt = get_prompt_for_code_generation(
        alpha_description=alpha_description,
        samples=samples
    )

    return prompt
