"""
Module: database.py
Description: Database setup, schema creation, and ORM models for coffee bean analysis.
"""

import os
import sqlite3
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from src.config import DATABASE_URL

# Create declarative base for ORM
Base = declarative_base()


class BeanAnalysis(Base):
    """Model for storing coffee bean analysis results."""
    
    __tablename__ = 'bean_analyses'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_name = Column(String(255), nullable=False)
    image_path = Column(String(500), nullable=False)
    output_path = Column(String(500), nullable=True)
    bean_count = Column(Integer, nullable=False)
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)
    processing_time = Column(Float, nullable=True)  # in seconds
    confidence = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<BeanAnalysis(id={self.id}, image={self.image_name}, count={self.bean_count})>"


class DatabaseManager:
    """Manages all database operations."""
    
    def __init__(self, database_url=None):
        """
        Initialize database connection.
        
        Args:
            database_url (str): Database URL. If None, uses config.DATABASE_URL
        """
        self.database_url = database_url or DATABASE_URL
        self.engine = None
        self.Session = None
        self._initialize()
    
    def _initialize(self):
        """Initialize database engine and create tables."""
        try:
            self.engine = create_engine(self.database_url, echo=False)
            self.Session = sessionmaker(bind=self.engine)
            
            # Create all tables
            Base.metadata.create_all(self.engine)
            print(f"✅ Database initialized successfully at: {self.database_url}")
        except Exception as e:
            print(f"❌ Error initializing database: {e}")
            raise
    
    def get_session(self):
        """Get a new database session."""
        if self.Session is None:
            self._initialize()
        return self.Session()
    
    def add_analysis(self, image_name, image_path, output_path, bean_count, 
                     width=None, height=None, processing_time=None, 
                     confidence=None, notes=None):
        """
        Add a new bean analysis record to the database.
        
        Args:
            image_name (str): Name of the image file
            image_path (str): Full path to the input image
            output_path (str): Full path to the output image
            bean_count (int): Number of beans detected
            width (int): Image width in pixels
            height (int): Image height in pixels
            processing_time (float): Time taken to process in seconds
            confidence (float): Confidence score (0-1)
            notes (str): Additional notes
            
        Returns:
            BeanAnalysis: The created database record
        """
        session = self.get_session()
        try:
            analysis = BeanAnalysis(
                image_name=image_name,
                image_path=image_path,
                output_path=output_path,
                bean_count=bean_count,
                image_width=width,
                image_height=height,
                processing_time=processing_time,
                confidence=confidence,
                notes=notes
            )
            session.add(analysis)
            session.commit()
            record_id = analysis.id
            print(f"✅ Analysis saved to database (ID: {record_id})")
            return analysis
        except Exception as e:
            session.rollback()
            print(f"❌ Error saving analysis to database: {e}")
            raise
        finally:
            session.close()
    
    def get_analysis(self, analysis_id):
        """
        Retrieve a specific analysis by ID.
        
        Args:
            analysis_id (int): ID of the analysis
            
        Returns:
            BeanAnalysis: The analysis record or None if not found
        """
        session = self.get_session()
        try:
            return session.query(BeanAnalysis).filter(BeanAnalysis.id == analysis_id).first()
        finally:
            session.close()
    
    def get_all_analyses(self):
        """
        Retrieve all analyses from the database.
        
        Returns:
            list: List of BeanAnalysis records
        """
        session = self.get_session()
        try:
            return session.query(BeanAnalysis).order_by(BeanAnalysis.created_at.desc()).all()
        finally:
            session.close()
    
    def get_analyses_by_image(self, image_name):
        """
        Retrieve all analyses for a specific image.
        
        Args:
            image_name (str): Name of the image file
            
        Returns:
            list: List of BeanAnalysis records
        """
        session = self.get_session()
        try:
            return session.query(BeanAnalysis).filter(
                BeanAnalysis.image_name == image_name
            ).order_by(BeanAnalysis.created_at.desc()).all()
        finally:
            session.close()
    
    def get_statistics(self):
        """
        Get statistics about all analyses.
        
        Returns:
            dict: Statistics including total analyses, average count, min/max counts
        """
        session = self.get_session()
        try:
            all_analyses = session.query(BeanAnalysis).all()
            if not all_analyses:
                return {
                    'total_analyses': 0,
                    'average_bean_count': 0,
                    'min_bean_count': 0,
                    'max_bean_count': 0,
                    'total_images_processed': 0
                }
            
            bean_counts = [a.bean_count for a in all_analyses]
            unique_images = len(set([a.image_name for a in all_analyses]))
            
            return {
                'total_analyses': len(all_analyses),
                'average_bean_count': sum(bean_counts) / len(bean_counts),
                'min_bean_count': min(bean_counts),
                'max_bean_count': max(bean_counts),
                'total_images_processed': unique_images
            }
        finally:
            session.close()
    
    def delete_analysis(self, analysis_id):
        """
        Delete an analysis from the database.
        
        Args:
            analysis_id (int): ID of the analysis to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        session = self.get_session()
        try:
            analysis = session.query(BeanAnalysis).filter(BeanAnalysis.id == analysis_id).first()
            if analysis:
                session.delete(analysis)
                session.commit()
                print(f"✅ Analysis {analysis_id} deleted from database")
                return True
            else:
                print(f"⚠️  Analysis {analysis_id} not found")
                return False
        except Exception as e:
            session.rollback()
            print(f"❌ Error deleting analysis: {e}")
            return False
        finally:
            session.close()
    
    def close(self):
        """Close the database connection."""
        if self.engine:
            self.engine.dispose()


# Global database manager instance
_db_manager = None


def get_db_manager():
    """Get or create the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
