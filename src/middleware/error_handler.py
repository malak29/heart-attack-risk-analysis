from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
import traceback
from datetime import datetime
from typing import Union

logger = logging.getLogger(__name__)

class ErrorHandler:
    """Centralized error handling"""
    
    @staticmethod
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions"""
        
        logger.error(f"HTTP error on {request.url.path}: {exc.detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.status_code,
                    "message": exc.detail,
                    "timestamp": datetime.now().isoformat(),
                    "path": str(request.url.path)
                }
            }
        )
    
    @staticmethod
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors"""
        
        logger.error(f"Validation error on {request.url.path}: {exc.errors()}")
        
        # Format validation errors
        errors = []
        for error in exc.errors():
            errors.append({
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })
        
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": 422,
                    "message": "Validation failed",
                    "details": errors,
                    "timestamp": datetime.now().isoformat(),
                    "path": str(request.url.path)
                }
            }
        )
    
    @staticmethod
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        
        # Log full traceback
        logger.error(f"Unhandled exception on {request.url.path}: {str(exc)}")
        logger.error(traceback.format_exc())
        
        # Don't expose internal errors in production
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": 500,
                    "message": "Internal server error",
                    "timestamp": datetime.now().isoformat(),
                    "path": str(request.url.path)
                }
            }
        )
    
    @staticmethod
    async def custom_exception_handler(request: Request, exc: Exception):
        """Handle custom application exceptions"""
        
        from src.exceptions import (
            DataValidationError,
            ModelNotFoundError,
            ModelTrainingError,
            PredictionError
        )
        
        status_code = 500
        message = str(exc)
        
        # Map custom exceptions to HTTP status codes
        if isinstance(exc, DataValidationError):
            status_code = 400
        elif isinstance(exc, ModelNotFoundError):
            status_code = 404
        elif isinstance(exc, ModelTrainingError):
            status_code = 500
        elif isinstance(exc, PredictionError):
            status_code = 500
        
        logger.error(f"Custom exception on {request.url.path}: {message}")
        
        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "code": status_code,
                    "message": message,
                    "type": exc.__class__.__name__,
                    "timestamp": datetime.now().isoformat(),
                    "path": str(request.url.path)
                }
            }
        )

def setup_exception_handlers(app):
    """Setup exception handlers for the FastAPI app"""
    
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException
    
    # HTTP exceptions
    app.add_exception_handler(HTTPException, ErrorHandler.http_exception_handler)
    
    # Validation errors
    app.add_exception_handler(RequestValidationError, ErrorHandler.validation_exception_handler)
    
    # General exceptions (catch-all)
    app.add_exception_handler(Exception, ErrorHandler.general_exception_handler)
    
    # Custom exceptions
    from src.exceptions import (
        DataValidationError,
        ModelNotFoundError,
        ModelTrainingError,
        PredictionError
    )
    
    app.add_exception_handler(DataValidationError, ErrorHandler.custom_exception_handler)
    app.add_exception_handler(ModelNotFoundError, ErrorHandler.custom_exception_handler)
    app.add_exception_handler(ModelTrainingError, ErrorHandler.custom_exception_handler)
    app.add_exception_handler(PredictionError, ErrorHandler.custom_exception_handler)