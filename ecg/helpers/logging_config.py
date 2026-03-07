"""Logfire configuration and initialization."""

import logfire
from pathlib import Path


def setup_logfire(
    service_name: str = "brugada-classifier",
    log_dir: Path = Path("./logs"),
    send_to_logfire: bool = False
) -> None:
    """
    Initialize Logfire for structured logging.
    
    Args:
        service_name: Name of the service for log identification
        log_dir: Directory for local log files
        send_to_logfire: Whether to send logs to Logfire cloud (requires API key)
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure Logfire
    logfire.configure(
        service_name=service_name,
        send_to_logfire=send_to_logfire,  # Set to True to send to cloud
        console=False,  # Log to console
    )
    
    logfire.info('Logfire initialized', service_name=service_name, log_dir=str(log_dir))


def log_data_statistics(stats: dict) -> None:
    """Log dataset statistics."""
    logfire.info(
        'Dataset statistics',
        total_samples=stats['total_samples'],
        normal_samples=stats['normal_samples'],
        brugada_samples=stats['brugada_samples'],
        class_ratio=stats['class_balance_ratio'],
        pos_weight=stats['pos_weight']
    )


def log_config(config_name: str, config: dict) -> None:
    """Log configuration."""
    logfire.info(f'{config_name} configuration loaded', **config)
