from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Financial Risk Control Intelligence"
    PROJECT_VERSION: str = "5.0.0"

    # Redis配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # 爬虫配置
    CRAWLER_TIMEOUT: int = 30
    CRAWLER_RETRY: int = 3

    # 模型配置
    CAUSAL_CONFIDENCE: float = 0.95
    MAX_PROPAGATION_DAYS: int = 7

    # API配置
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # 风控增强配置
    VAR_FUSION_WEIGHTS: str = "0.4,0.3,0.3"  # 历史法/参数法/蒙特卡洛 权重
    STRESS_TEST_MC_SIMS: int = 10000
    STRESS_TEST_HORIZON_DAYS: int = 30
    SINGLE_ASSET_LIMIT: float = 0.10
    SECTOR_EXPOSURE_LIMIT: float = 0.30
    LIQUIDITY_LCR_MINIMUM: float = 0.80

    # DeepSeek API 配置
    DEEPSEEK_API_KEY: str = ""

    # 数据源配置
    KLINE_PRIMARY_SOURCE: str = "tencent"  # OHLCV主数据源: tencent/eastmoney
    KLINE_FALLBACK_SOURCE: str = "eastmoney"  # OHLCV备选数据源
    FINANCIAL_SOURCE: str = "eastmoney"  # 财务数据源: eastmoney
    MARKET_BENCHMARK_SOURCE: str = "sina"  # 市场基准数据源: sina/eastmoney
    
    class Config:
        env_file = ".env"

settings = Settings()
