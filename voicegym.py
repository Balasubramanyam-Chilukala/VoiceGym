import cv2
import mediapipe as mp
import numpy as np
import time
import requests
import google.generativeai as genai
import pygame
import os
import json
from threading import Thread
import tempfile
from dotenv import load_dotenv
import math
import random
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import stat
from collections import deque
import hashlib
import psycopg2
from psycopg2.extras import RealDictCursor
import psycopg2.pool
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import base64
from io import BytesIO
import uuid

load_dotenv()
print("🏋️ ULTIMATE ADVANCED VOICEGYM - Revolutionary AI Fitness Ecosystem Loading...")

# SETUP

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MURF_API_KEY = os.getenv("MURF_API_KEY")

# PostgreSQL Configuration (hidden in backend)
POSTGRES_CONFIG = {
    'host': os.getenv("POSTGRES_HOST", "localhost"),
    'port': os.getenv("POSTGRES_PORT", "5432"),
    'database': os.getenv("POSTGRES_DB", "voicegym"),
    'user': os.getenv("POSTGRES_USER", "postgres"),
    'password': os.getenv("POSTGRES_PASSWORD", "password")
}

if not GEMINI_API_KEY or not MURF_API_KEY:
    print("❌ Please add your actual API keys to the .env file!")
    raise SystemExit()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
print("✅ API Keys configured!")

# Current user info
CURRENT_USER = "Balasubramanyam-Chilukala"
CURRENT_DATE = "2025-08-30"
CURRENT_TIME = "03:34:42"

# PROGRESS DATABASE MANAGER

class ProgressDatabaseManager:
    """Complete progress tracking with achievements and social features"""
    
    def __init__(self):
        self.connection_pool = None
        self.init_connection_pool()
        self.init_database()
        
    def init_connection_pool(self):
        """Initialize connection pool"""
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=20,
                host=POSTGRES_CONFIG['host'],
                port=POSTGRES_CONFIG['port'],
                database=POSTGRES_CONFIG['database'],
                user=POSTGRES_CONFIG['user'],
                password=POSTGRES_CONFIG['password']
            )
            print("✅ Database connection established!")
        except Exception as e:
            print(f"❌ Failed to connect to database: {e}")
            print("🔧 Make sure database is running and credentials are correct in .env file")
            print("📝 Required .env variables:")
            print("   POSTGRES_HOST=localhost")
            print("   POSTGRES_PORT=5432") 
            print("   POSTGRES_DB=voicegym")
            print("   POSTGRES_USER=postgres")
            print("   POSTGRES_PASSWORD=your_password")
            raise SystemExit()
    
    def get_connection(self):
        """Get connection from pool"""
        return self.connection_pool.getconn()
    
    def return_connection(self, conn):
        """Return connection to pool"""
        self.connection_pool.putconn(conn)
    
    def init_database(self):
        """Initialize all database tables"""
        conn = self.get_connection()
        
        try:
            cursor = conn.cursor()
            
            # Enable UUID extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    username VARCHAR(255) UNIQUE NOT NULL,
                    created_date DATE NOT NULL DEFAULT CURRENT_DATE,
                    age_group VARCHAR(50) DEFAULT 'adult',
                    fitness_level VARCHAR(50) DEFAULT 'beginner',
                    total_workouts INTEGER DEFAULT 0,
                    total_reps INTEGER DEFAULT 0,
                    current_streak INTEGER DEFAULT 0,
                    longest_streak INTEGER DEFAULT 0,
                    last_workout_date DATE,
                    total_workout_minutes DECIMAL(10,2) DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Workouts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS workouts (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                    workout_date DATE NOT NULL DEFAULT CURRENT_DATE,
                    duration_minutes DECIMAL(10,2) NOT NULL,
                    total_reps INTEGER NOT NULL,
                    exercises_performed JSONB NOT NULL,
                    average_form_score DECIMAL(3,1) NOT NULL,
                    perfect_reps INTEGER DEFAULT 0,
                    exercise_switches INTEGER DEFAULT 0,
                    ai_feedback_calls INTEGER DEFAULT 0,
                    form_corrections INTEGER DEFAULT 0,
                    injury_warnings INTEGER DEFAULT 0,
                    voice_coach VARCHAR(255) NOT NULL,
                    language VARCHAR(50) NOT NULL,
                    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_id VARCHAR(100),
                    workout_rating INTEGER CHECK (workout_rating >= 1 AND workout_rating <= 10)
                )
            ''')
            
            # Exercise records
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS exercise_records (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                    exercise_name VARCHAR(255) NOT NULL,
                    max_reps INTEGER DEFAULT 0,
                    best_form_score DECIMAL(3,1) DEFAULT 0.0,
                    total_attempts INTEGER DEFAULT 0,
                    last_performed DATE,
                    performance_history JSONB DEFAULT '[]'::jsonb,
                    form_improvements TEXT[],
                    personal_records JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, exercise_name)
                )
            ''')
            
            # Achievements table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS achievements (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                    achievement_type VARCHAR(100) NOT NULL,
                    achievement_name VARCHAR(255) NOT NULL,
                    description TEXT NOT NULL,
                    unlocked_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    badge_level VARCHAR(50) DEFAULT 'bronze',
                    points_awarded INTEGER DEFAULT 0,
                    requirements_met JSONB,
                    category VARCHAR(100),
                    rarity VARCHAR(50) DEFAULT 'common'
                )
            ''')
            
            # Family groups
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS family_groups (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    group_name VARCHAR(255) NOT NULL,
                    created_by UUID REFERENCES users(id),
                    created_date DATE NOT NULL DEFAULT CURRENT_DATE,
                    total_members INTEGER DEFAULT 1,
                    group_total_reps INTEGER DEFAULT 0,
                    group_settings JSONB DEFAULT '{}'::jsonb,
                    is_active BOOLEAN DEFAULT TRUE,
                    group_goals JSONB DEFAULT '[]'::jsonb
                )
            ''')
            
            # Family members
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS family_members (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    group_id UUID REFERENCES family_groups(id) ON DELETE CASCADE,
                    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                    relationship VARCHAR(100) DEFAULT 'member',
                    joined_date DATE NOT NULL DEFAULT CURRENT_DATE,
                    role VARCHAR(50) DEFAULT 'member',
                    permissions TEXT[],
                    member_goals JSONB DEFAULT '[]'::jsonb,
                    UNIQUE(group_id, user_id)
                )
            ''')
            
            # Global challenges
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS global_challenges (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    challenge_name VARCHAR(255) NOT NULL,
                    challenge_type VARCHAR(100) NOT NULL,
                    target_value BIGINT NOT NULL,
                    current_value BIGINT DEFAULT 0,
                    start_date DATE NOT NULL,
                    end_date DATE NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    challenge_description TEXT,
                    reward_points INTEGER DEFAULT 0,
                    difficulty_level VARCHAR(50) DEFAULT 'medium',
                    participating_countries TEXT[],
                    challenge_rules JSONB DEFAULT '{}'::jsonb
                )
            ''')
            
            # Challenge participation
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS challenge_participation (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    challenge_id UUID REFERENCES global_challenges(id) ON DELETE CASCADE,
                    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                    contribution BIGINT DEFAULT 0,
                    joined_date DATE NOT NULL DEFAULT CURRENT_DATE,
                    last_contribution_date DATE,
                    daily_contributions JSONB DEFAULT '{}'::jsonb,
                    rank_position INTEGER,
                    points_earned INTEGER DEFAULT 0,
                    UNIQUE(challenge_id, user_id)
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_workouts_user_date ON workouts(user_id, workout_date);')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_workouts_exercises ON workouts USING GIN(exercises_performed);')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_achievements_user ON achievements(user_id, unlocked_date);')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_challenges_active ON global_challenges(is_active, end_date);')
            
            # Create trigger for updating updated_at
            cursor.execute('''
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
            ''')
            
            cursor.execute('''
                DROP TRIGGER IF EXISTS update_users_updated_at ON users;
                CREATE TRIGGER update_users_updated_at 
                    BEFORE UPDATE ON users 
                    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
            ''')
            
            conn.commit()
            print("✅ Database schema initialized with advanced features!")
            
        except Exception as e:
            print(f"❌ Database initialization error: {e}")
            conn.rollback()
            raise
        finally:
            self.return_connection(conn)
    
    def get_or_create_user(self, username):
        """Get existing user or create new one"""
        conn = self.get_connection()
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
            
            if not user:
                cursor.execute('''
                    INSERT INTO users (username, last_workout_date)
                    VALUES (%s, %s)
                    RETURNING id
                ''', (username, CURRENT_DATE))
                
                user_result = cursor.fetchone()
                user_id = user_result['id']
                
                # Create default exercise records
                exercises = ['bicep_curls', 'push_ups', 'squats', 'shoulder_press', 'plank', 'lunges']
                for exercise in exercises:
                    cursor.execute('''
                        INSERT INTO exercise_records (user_id, exercise_name, last_performed)
                        VALUES (%s, %s, %s)
                    ''', (user_id, exercise, CURRENT_DATE))
                
                conn.commit()
                print(f"✅ New user created: {username}")
            else:
                user_id = user['id']
            
            return user_id
            
        except Exception as e:
            print(f"❌ User creation error: {e}")
            conn.rollback()
            raise
        finally:
            self.return_connection(conn)
    
    def save_workout(self, user_id, workout_data):
        """Save complete workout session"""
        conn = self.get_connection()
        
        try:
            cursor = conn.cursor()
            
            # Insert workout with JSONB data
            cursor.execute('''
                INSERT INTO workouts (
                    user_id, workout_date, duration_minutes, total_reps,
                    exercises_performed, average_form_score, perfect_reps,
                    exercise_switches, ai_feedback_calls, form_corrections,
                    injury_warnings, voice_coach, language, session_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            ''', (
                user_id, CURRENT_DATE, workout_data['duration'],
                workout_data['total_reps'], json.dumps(workout_data['exercises']),
                workout_data['avg_form_score'], workout_data['perfect_reps'],
                workout_data['exercise_switches'], workout_data['ai_calls'],
                workout_data['form_corrections'], workout_data['injury_warnings'],
                workout_data['voice_coach'], workout_data['language'],
                str(uuid.uuid4())
            ))
            
            workout_id = cursor.fetchone()[0]
            
            # Update user stats
            cursor.execute('''
                UPDATE users SET 
                    total_workouts = total_workouts + 1,
                    total_reps = total_reps + %s,
                    total_workout_minutes = total_workout_minutes + %s,
                    last_workout_date = %s
                WHERE id = %s
            ''', (workout_data['total_reps'], workout_data['duration'], CURRENT_DATE, user_id))
            
            # Update exercise records with JSONB operations
            for exercise_name, exercise_stats in workout_data['exercises'].items():
                cursor.execute('''
                    UPDATE exercise_records SET
                        max_reps = GREATEST(max_reps, %s),
                        best_form_score = GREATEST(best_form_score, %s),
                        total_attempts = total_attempts + 1,
                        last_performed = %s,
                        performance_history = performance_history || %s::jsonb
                    WHERE user_id = %s AND exercise_name = %s
                ''', (
                    exercise_stats['reps'], exercise_stats['best_form'],
                    CURRENT_DATE, 
                    json.dumps({
                        'date': CURRENT_DATE,
                        'reps': exercise_stats['reps'],
                        'form_score': exercise_stats['best_form'],
                        'duration': workout_data['duration']
                    }),
                    user_id, exercise_name
                ))
            
            conn.commit()
            
            # Check for new achievements
            new_achievements = self.check_achievements(user_id)
            
            print(f"✅ Workout saved! ID: {workout_id}")
            return workout_id, new_achievements
            
        except Exception as e:
            print(f"❌ Workout save error: {e}")
            conn.rollback()
            raise
        finally:
            self.return_connection(conn)
    
    def update_streak(self, user_id):
        """Calculate and update workout streak"""
        conn = self.get_connection()
        
        try:
            cursor = conn.cursor()
            
            # Get user's last workout date
            cursor.execute('''
                SELECT last_workout_date, current_streak 
                FROM users WHERE id = %s
            ''', (user_id,))
            
            result = cursor.fetchone()
            
            if result:
                last_workout_date, current_streak = result
                today = datetime.strptime(CURRENT_DATE, "%Y-%m-%d").date()
                
                if last_workout_date:
                    days_diff = (today - last_workout_date).days
                    
                    if days_diff == 1:  # Consecutive day
                        new_streak = current_streak + 1
                    elif days_diff == 0:  # Same day
                        new_streak = current_streak
                    else:  # Streak broken
                        new_streak = 1
                else:
                    new_streak = 1
                
                # Update streak
                cursor.execute('''
                    UPDATE users SET 
                        current_streak = %s,
                        longest_streak = GREATEST(longest_streak, %s)
                    WHERE id = %s
                ''', (new_streak, new_streak, user_id))
                
                conn.commit()
                return new_streak
            
            return 1
            
        except Exception as e:
            print(f"❌ Streak update error: {e}")
            conn.rollback()
            raise
        finally:
            self.return_connection(conn)
    
    def check_achievements(self, user_id):
        """Check and unlock new achievements"""
        conn = self.get_connection()
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get user stats
            cursor.execute('''
                SELECT total_workouts, total_reps, current_streak, longest_streak,
                       total_workout_minutes FROM users WHERE id = %s
            ''', (user_id,))
            stats = cursor.fetchone()
            
            if not stats:
                return []
            
            # Get existing achievements
            cursor.execute('''
                SELECT achievement_name FROM achievements WHERE user_id = %s
            ''', (user_id,))
            existing = {row['achievement_name'] for row in cursor.fetchall()}
            
            new_achievements = []
            
            # Define achievement milestones
            achievement_types = [
                {
                    'type': 'workouts',
                    'field': 'total_workouts',
                    'milestones': [
                        (1, "First Workout", "Completed your first VoiceGym session!", 10),
                        (5, "Getting Started", "5 workouts completed - building the habit!", 25),
                        (10, "Consistency Builder", "10 workouts done - you're committed!", 50),
                        (25, "Fitness Enthusiast", "25 workouts completed - amazing dedication!", 100),
                        (50, "VoiceGym Veteran", "50 workouts - you're a fitness warrior!", 200),
                        (100, "Century Club", "100 workouts achieved - legendary status!", 500)
                    ]
                },
                {
                    'type': 'reps',
                    'field': 'total_reps',
                    'milestones': [
                        (100, "Rep Rookie", "First 100 reps completed!", 15),
                        (500, "Rep Warrior", "500 reps achieved - building strength!", 50),
                        (1000, "Rep Master", "1000 reps milestone - incredible effort!", 100),
                        (2500, "Rep Legend", "2500 reps completed - unstoppable!", 250),
                        (5000, "Rep Titan", "5000 reps achieved - ultimate dedication!", 500)
                    ]
                },
                {
                    'type': 'streaks',
                    'field': 'longest_streak',
                    'milestones': [
                        (3, "Three Day Streak", "3 consecutive workout days!", 30),
                        (7, "Weekly Warrior", "7 day workout streak - amazing!", 70),
                        (14, "Two Week Champion", "14 day streak - incredible consistency!", 140),
                        (30, "Monthly Master", "30 day streak - you're unstoppable!", 300),
                        (60, "Streak Superhero", "60 day streak - legendary dedication!", 600),
                        (100, "Streak Titan", "100 day streak - absolute champion!", 1000)
                    ]
                }
            ]
            
            for achievement_type in achievement_types:
                field_value = stats[achievement_type['field']]
                
                for milestone, name, desc, points in achievement_type['milestones']:
                    if field_value >= milestone and name not in existing:
                        self._unlock_achievement(cursor, user_id, achievement_type['type'], name, desc, points, {
                            'milestone': milestone,
                            'current_value': field_value,
                            'achievement_date': CURRENT_DATE
                        })
                        new_achievements.append(name)
            
            # Time-based achievements
            total_minutes = float(stats['total_workout_minutes'])
            time_achievements = [
                (60, "Hour Hero", "60 minutes of total workout time!", 60),
                (300, "Five Hour Fighter", "5 hours of total workout time!", 300),
                (600, "Ten Hour Titan", "10 hours of total workout time!", 600),
                (1200, "Twenty Hour Legend", "20 hours of total workout time!", 1200)
            ]
            
            for minutes, name, desc, points in time_achievements:
                if total_minutes >= minutes and name not in existing:
                    self._unlock_achievement(cursor, user_id, 'time', name, desc, points, {
                        'minutes_milestone': minutes,
                        'current_minutes': total_minutes
                    })
                    new_achievements.append(name)
            
            conn.commit()
            return new_achievements
            
        except Exception as e:
            print(f"❌ Achievement check error: {e}")
            conn.rollback()
            return []
        finally:
            self.return_connection(conn)
    
    def _unlock_achievement(self, cursor, user_id, achievement_type, name, description, points, requirements):
        """Helper to unlock achievement"""
        cursor.execute('''
            INSERT INTO achievements (
                user_id, achievement_type, achievement_name, description, 
                points_awarded, requirements_met, category
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''', (
            user_id, achievement_type, name, description, 
            points, json.dumps(requirements), 'fitness'
        ))
    
    def get_user_stats(self, user_id):
        """Get comprehensive user statistics"""
        conn = self.get_connection()
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Basic user stats
            cursor.execute('''
                SELECT username, total_workouts, total_reps, current_streak,
                       longest_streak, total_workout_minutes, created_date
                FROM users WHERE id = %s
            ''', (user_id,))
            user_stats = cursor.fetchone()
            
            # Recent workouts with JSONB queries
            cursor.execute('''
                SELECT workout_date, duration_minutes, total_reps, 
                       average_form_score, exercises_performed
                FROM workouts 
                WHERE user_id = %s 
                ORDER BY workout_date DESC 
                LIMIT 10
            ''', (user_id,))
            recent_workouts = cursor.fetchall()
            
            # Exercise records with performance history
            cursor.execute('''
                SELECT exercise_name, max_reps, best_form_score, 
                       total_attempts, performance_history
                FROM exercise_records 
                WHERE user_id = %s
            ''', (user_id,))
            exercise_records = cursor.fetchall()
            
            # Achievements with points
            cursor.execute('''
                SELECT achievement_name, description, unlocked_date, 
                       achievement_type, points_awarded, requirements_met
                FROM achievements 
                WHERE user_id = %s 
                ORDER BY unlocked_date DESC
            ''', (user_id,))
            achievements = cursor.fetchall()
            
            # Global ranking
            cursor.execute('''
                SELECT COUNT(*) + 1 as global_rank
                FROM users 
                WHERE total_reps > (SELECT total_reps FROM users WHERE id = %s)
            ''', (user_id,))
            global_rank = cursor.fetchone()['global_rank']
            
            return {
                'user': dict(user_stats) if user_stats else None,
                'recent_workouts': [dict(w) for w in recent_workouts],
                'exercise_records': [dict(e) for e in exercise_records],
                'achievements': [dict(a) for a in achievements],
                'global_rank': global_rank
            }
            
        except Exception as e:
            print(f"❌ Stats retrieval error: {e}")
            return {}
        finally:
            self.return_connection(conn)
    
    def generate_progress_chart(self, user_id):
        """Generate workout progress visualization"""
        conn = self.get_connection()
        
        try:
            cursor = conn.cursor()
            
            # Get workout history
            cursor.execute('''
                SELECT workout_date, total_reps, average_form_score, duration_minutes
                FROM workouts 
                WHERE user_id = %s 
                ORDER BY workout_date
            ''', (user_id,))
            workouts = cursor.fetchall()
            
            if len(workouts) < 2:
                return None
            
            # Prepare data
            dates = [w[0] for w in workouts]
            reps = [w[1] for w in workouts]
            form_scores = [float(w[2]) for w in workouts]
            durations = [float(w[3]) for w in workouts]
            
            # Create enhanced chart
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'VoiceGym Progress Report - {CURRENT_USER}', fontsize=16, fontweight='bold')
            
            # Reps over time
            ax1.plot(dates, reps, 'b-o', linewidth=2, markersize=6)
            ax1.set_title('Reps per Workout', fontweight='bold')
            ax1.set_ylabel('Total Reps')
            ax1.grid(True, alpha=0.3)
            
            # Form scores over time
            ax2.plot(dates, form_scores, 'g-o', linewidth=2, markersize=6)
            ax2.set_title('Form Score Progress', fontweight='bold')
            ax2.set_ylabel('Average Form Score')
            ax2.set_ylim(0, 10)
            ax2.grid(True, alpha=0.3)
            
            # Workout duration
            ax3.plot(dates, durations, 'r-o', linewidth=2, markersize=6)
            ax3.set_title('Workout Duration', fontweight='bold')
            ax3.set_ylabel('Minutes')
            ax3.grid(True, alpha=0.3)
            
            # Weekly summary
            cursor.execute('''
                SELECT DATE_TRUNC('week', workout_date) as week_start,
                       SUM(total_reps) as weekly_reps,
                       COUNT(*) as workout_count
                FROM workouts 
                WHERE user_id = %s
                GROUP BY DATE_TRUNC('week', workout_date)
                ORDER BY week_start
            ''', (user_id,))
            
            weekly_data = cursor.fetchall()
            
            if weekly_data:
                week_labels = [f'Week {i+1}' for i in range(len(weekly_data))]
                weekly_reps = [w[1] for w in weekly_data]
                
                ax4.bar(week_labels, weekly_reps, color='purple', alpha=0.7)
                ax4.set_title('Weekly Rep Totals', fontweight='bold')
                ax4.set_ylabel('Total Reps')
            else:
                ax4.text(0.5, 0.5, 'More data needed\nfor weekly view', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax4.transAxes, fontsize=12)
                ax4.set_title('Weekly Summary (Coming Soon)', fontweight='bold')
            
            # Format x-axis
            for ax in [ax1, ax2, ax3]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//5)))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = f"progress_chart_{str(user_id)[:8]}_{int(time.time())}.png"
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            print(f"❌ Chart generation error: {e}")
            return None
        finally:
            self.return_connection(conn)
    
    def contribute_to_global_challenges(self, user_id, workout_data):
        """Add user's workout to global challenges"""
        conn = self.get_connection()
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get active challenges
            cursor.execute('''
                SELECT * FROM global_challenges 
                WHERE is_active = TRUE AND end_date >= CURRENT_DATE
            ''')
            challenges = cursor.fetchall()
            
            contributions = []
            
            for challenge in challenges:
                contribution = 0
                
                if challenge['challenge_type'] == 'total_reps':
                    contribution = workout_data['total_reps']
                elif challenge['challenge_type'] in workout_data['exercises']:
                    contribution = workout_data['exercises'][challenge['challenge_type']]['reps']
                
                if contribution > 0:
                    # Update challenge progress atomically
                    cursor.execute('''
                        UPDATE global_challenges 
                        SET current_value = current_value + %s 
                        WHERE id = %s
                    ''', (contribution, challenge['id']))
                    
                    # Upsert user participation
                    cursor.execute('''
                        INSERT INTO challenge_participation 
                        (challenge_id, user_id, contribution, last_contribution_date, daily_contributions)
                        VALUES (%s, %s, %s, CURRENT_DATE, %s)
                        ON CONFLICT (challenge_id, user_id) 
                        DO UPDATE SET 
                            contribution = challenge_participation.contribution + %s,
                            last_contribution_date = CURRENT_DATE,
                            daily_contributions = challenge_participation.daily_contributions || %s
                    ''', (
                        challenge['id'], user_id, contribution, 
                        json.dumps({CURRENT_DATE: contribution}),
                        contribution,
                        json.dumps({CURRENT_DATE: contribution})
                    ))
                    
                    contributions.append({
                        'challenge': challenge['challenge_name'],
                        'contribution': contribution,
                        'new_total': challenge['current_value'] + contribution,
                        'target': challenge['target_value']
                    })
            
            conn.commit()
            return contributions
            
        except Exception as e:
            print(f"❌ Global challenge contribution error: {e}")
            conn.rollback()
            return []
        finally:
            self.return_connection(conn)
    
    def get_global_leaderboard(self, challenge_id=None, limit=10):
        """Get global challenge leaderboard"""
        conn = self.get_connection()
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            if challenge_id:
                cursor.execute('''
                    SELECT u.username, cp.contribution, u.total_reps,
                           RANK() OVER (ORDER BY cp.contribution DESC) as rank
                    FROM challenge_participation cp
                    JOIN users u ON cp.user_id = u.id
                    WHERE cp.challenge_id = %s
                    ORDER BY cp.contribution DESC
                    LIMIT %s
                ''', (challenge_id, limit))
            else:
                cursor.execute('''
                    SELECT username, total_reps, total_workouts, current_streak,
                           RANK() OVER (ORDER BY total_reps DESC) as rank
                    FROM users
                    ORDER BY total_reps DESC
                    LIMIT %s
                ''', (limit,))
            
            leaderboard = cursor.fetchall()
            return [dict(row) for row in leaderboard]
            
        except Exception as e:
            print(f"❌ Leaderboard error: {e}")
            return []
        finally:
            self.return_connection(conn)
    
    def close_pool(self):
        """Close the connection pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            print("✅ Database connection pool closed")

# EXERCISE DEFINITIONS & BIOMECHANICS

class ExerciseDefinitions:
    """Complete exercise database with biomechanics and form analysis"""
    
    @staticmethod
    def get_exercises():
        return {
            'bicep_curls': {
                'name': 'Bicep Curls',
                'landmarks': [11, 13, 15],
                'down_threshold': 160,
                'up_threshold': 50,
                'detection_landmarks': [11, 12, 13, 14, 15, 16],
                'form_points': {
                    'elbow_stability': [11, 13],
                    'wrist_alignment': [13, 15],
                    'shoulder_position': [11, 12]
                }
            },
            'push_ups': {
                'name': 'Push-ups',
                'landmarks': [11, 13, 15, 23, 25, 27],
                'down_threshold': 90,
                'up_threshold': 160,
                'detection_landmarks': [11, 12, 13, 14, 15, 16, 23, 24],
                'form_points': {
                    'body_alignment': [11, 23, 25],
                    'elbow_angle': [11, 13, 15],
                    'core_stability': [11, 23, 27]
                }
            },
            'squats': {
                'name': 'Squats',
                'landmarks': [23, 25, 27],
                'down_threshold': 90,
                'up_threshold': 160,
                'detection_landmarks': [11, 12, 23, 24, 25, 26, 27, 28],
                'form_points': {
                    'knee_alignment': [23, 25, 27],
                    'back_straight': [11, 23, 25],
                    'depth_check': [23, 25]
                }
            },
            'lunges': {
                'name': 'Lunges',
                'landmarks': [23, 25, 27],
                'down_threshold': 90,
                'up_threshold': 160,
                'detection_landmarks': [23, 24, 25, 26, 27, 28],
                'form_points': {
                    'front_knee': [23, 25, 27],
                    'back_leg': [24, 26, 28],
                    'balance': [11, 23]
                }
            },
            'shoulder_press': {
                'name': 'Shoulder Press',
                'landmarks': [11, 13, 15],
                'down_threshold': 90,
                'up_threshold': 170,
                'detection_landmarks': [11, 12, 13, 14, 15, 16],
                'form_points': {
                    'shoulder_stability': [11, 12],
                    'elbow_tracking': [11, 13, 15],
                    'wrist_alignment': [13, 15]
                }
            },
            'plank': {
                'name': 'Plank Hold',
                'landmarks': [11, 23, 27],
                'down_threshold': 160,
                'up_threshold': 200,
                'detection_landmarks': [11, 12, 13, 14, 15, 16, 23, 24, 25, 26],
                'form_points': {
                    'body_line': [11, 23, 27],
                    'core_engagement': [11, 23],
                    'shoulder_position': [11, 13, 15]
                }
            },
            'jumping_jacks': {
                'name': 'Jumping Jacks',
                'landmarks': [11, 15, 23, 27],
                'down_threshold': 45,
                'up_threshold': 160,
                'detection_landmarks': [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28],
                'form_points': {
                    'arm_coordination': [11, 15, 12, 16],
                    'leg_coordination': [23, 27, 24, 28],
                    'timing': [11, 23]
                }
            },
            'mountain_climbers': {
                'name': 'Mountain Climbers',
                'landmarks': [23, 25, 27],
                'down_threshold': 90,
                'up_threshold': 160,
                'detection_landmarks': [11, 12, 15, 16, 23, 24, 25, 26, 27, 28],
                'form_points': {
                    'plank_position': [11, 23, 27],
                    'knee_drive': [23, 25],
                    'core_stability': [11, 23]
                }
            }
        }

# SMART EXERCISE DETECTION SYSTEM

class SmartExerciseDetector:
    """AI-powered exercise detection using pose landmarks"""
    
    def __init__(self):
        self.exercises = ExerciseDefinitions.get_exercises()
        self.detection_history = deque(maxlen=30)
        self.current_exercise = 'bicep_curls'
        self.confidence_threshold = 0.7
        self.detection_stability = 15
        
    def detect_exercise(self, landmarks):
        """Detect which exercise user is performing"""
        if not landmarks:
            return self.current_exercise
        
        exercise_scores = {}
        
        for exercise_name, exercise_data in self.exercises.items():
            score = self._calculate_exercise_score(landmarks, exercise_data)
            exercise_scores[exercise_name] = score
        
        best_exercise = max(exercise_scores, key=exercise_scores.get)
        best_score = exercise_scores[best_exercise]
        
        self.detection_history.append((best_exercise, best_score))
        
        if len(self.detection_history) >= self.detection_stability:
            recent_detections = [det[0] for det in list(self.detection_history)[-self.detection_stability:]]
            most_common = max(set(recent_detections), key=recent_detections.count)
            
            if (recent_detections.count(most_common) >= self.detection_stability * 0.8 and 
                most_common != self.current_exercise):
                print(f"🔄 Exercise change detected: {self.current_exercise} → {most_common}")
                self.current_exercise = most_common
        
        return self.current_exercise
    
    def _calculate_exercise_score(self, landmarks, exercise_data):
        """Calculate how well pose matches exercise pattern"""
        score = 0.0
        detection_landmarks = exercise_data['detection_landmarks']
        
        visible_landmarks = 0
        for idx in detection_landmarks:
            if idx < len(landmarks) and landmarks[idx].visibility > 0.5:
                visible_landmarks += 1
        
        visibility_score = visible_landmarks / len(detection_landmarks)
        score += visibility_score * 0.3
        
        if exercise_data['name'] == 'Bicep Curls':
            score += self._detect_bicep_curls(landmarks) * 0.7
        elif exercise_data['name'] == 'Push-ups':
            score += self._detect_push_ups(landmarks) * 0.7
        elif exercise_data['name'] == 'Squats':
            score += self._detect_squats(landmarks) * 0.7
        elif exercise_data['name'] == 'Shoulder Press':
            score += self._detect_shoulder_press(landmarks) * 0.7
        elif exercise_data['name'] == 'Plank Hold':
            score += self._detect_plank(landmarks) * 0.7
        else:
            score += 0.5
        
        return min(score, 1.0)
    
    def _detect_bicep_curls(self, landmarks):
        try:
            if len(landmarks) > 24:
                shoulder = landmarks[11]
                hip = landmarks[23]
                if abs(shoulder.x - hip.x) > 0.1:
                    return 0.3
            
            if len(landmarks) > 15:
                shoulder = landmarks[11]
                elbow = landmarks[13]
                wrist = landmarks[15]
                
                if abs(elbow.x - shoulder.x) > 0.2:
                    return 0.4
                
                arm_vertical_motion = abs(wrist.y - elbow.y)
                if arm_vertical_motion > 0.1:
                    return 0.8
            
            return 0.6
        except:
            return 0.0
    
    def _detect_push_ups(self, landmarks):
        try:
            if len(landmarks) > 27:
                shoulder = landmarks[11]
                hip = landmarks[23]
                ankle = landmarks[27]
                
                body_slope = abs(shoulder.y - ankle.y)
                if body_slope < 0.3:
                    return 0.8
                
            return 0.3
        except:
            return 0.0
    
    def _detect_squats(self, landmarks):
        try:
            if len(landmarks) > 27:
                hip = landmarks[23]
                knee = landmarks[25]
                ankle = landmarks[27]
                
                knee_angle = self._calculate_angle_from_points(hip, knee, ankle)
                if 70 < knee_angle < 150:
                    return 0.8
                
            return 0.4
        except:
            return 0.0
    
    def _detect_shoulder_press(self, landmarks):
        try:
            if len(landmarks) > 15:
                shoulder = landmarks[11]
                elbow = landmarks[13]
                wrist = landmarks[15]
                
                if wrist.y < shoulder.y and elbow.y < shoulder.y:
                    return 0.8
                
            return 0.3
        except:
            return 0.0
    
    def _detect_plank(self, landmarks):
        try:
            if len(landmarks) > 27:
                shoulder = landmarks[11]
                hip = landmarks[23]
                ankle = landmarks[27]
                
                body_line_score = 1.0 - abs(shoulder.y - hip.y) - abs(hip.y - ankle.y)
                if body_line_score > 0.6:
                    return 0.9
                
            return 0.2
        except:
            return 0.0
    
    def _calculate_angle_from_points(self, a, b, c):
        try:
            radians = np.arctan2(c.y-b.y, c.x-b.x) - np.arctan2(a.y-b.y, a.x-b.x)
            angle = np.abs(radians * 180.0 / np.pi)
            return 360 - angle if angle > 180 else angle
        except:
            return 90

# ADVANCED FORM ANALYSIS SYSTEM

class AdvancedFormAnalyzer:
    """Real-time form analysis with biomechanical feedback"""
    
    def __init__(self):
        self.form_history = deque(maxlen=60)
        self.injury_patterns = deque(maxlen=30)
        self.exercises = ExerciseDefinitions.get_exercises()
        
    def analyze_form(self, landmarks, exercise_name, current_angle):
        """Comprehensive form analysis"""
        if not landmarks or exercise_name not in self.exercises:
            return self._default_form_score()
        
        exercise_data = self.exercises[exercise_name]
        form_analysis = {
            'overall_score': 5.0,
            'feedback': [],
            'warnings': [],
            'corrections': [],
            'injury_risk': 'low'
        }
        
        if exercise_name == 'bicep_curls':
            form_analysis = self._analyze_bicep_form(landmarks, current_angle, form_analysis)
        elif exercise_name == 'push_ups':
            form_analysis = self._analyze_pushup_form(landmarks, current_angle, form_analysis)
        elif exercise_name == 'squats':
            form_analysis = self._analyze_squat_form(landmarks, current_angle, form_analysis)
        elif exercise_name == 'shoulder_press':
            form_analysis = self._analyze_shoulder_press_form(landmarks, current_angle, form_analysis)
        elif exercise_name == 'plank':
            form_analysis = self._analyze_plank_form(landmarks, current_angle, form_analysis)
        
        self.form_history.append(form_analysis)
        return form_analysis
    
    def _analyze_bicep_form(self, landmarks, angle, analysis):
        try:
            shoulder = landmarks[11]
            elbow = landmarks[13]
            wrist = landmarks[15]
            
            elbow_drift = abs(elbow.x - shoulder.x)
            if elbow_drift > 0.15:
                analysis['feedback'].append("Keep your elbow closer to your body")
                analysis['overall_score'] -= 1.0
            
            wrist_angle = abs(wrist.x - elbow.x)
            if wrist_angle > 0.1:
                analysis['corrections'].append("Align your wrist with your elbow")
                analysis['overall_score'] -= 0.5
            
            if angle > 170:
                analysis['feedback'].append("Great extension! Feel the stretch")
                analysis['overall_score'] += 1.0
            elif angle < 30:
                analysis['feedback'].append("Perfect contraction! Squeeze those biceps")
                analysis['overall_score'] += 1.0
            
            if elbow_drift > 0.2:
                analysis['warnings'].append("⚠️ Elbow flaring - risk of shoulder strain")
                analysis['injury_risk'] = 'medium'
                
        except Exception as e:
            pass
        
        return analysis
    
    def _analyze_pushup_form(self, landmarks, angle, analysis):
        try:
            shoulder = landmarks[11]
            elbow = landmarks[13]
            wrist = landmarks[15]
            hip = landmarks[23]
            knee = landmarks[25]
            
            body_line = self._calculate_body_line([shoulder, hip, knee])
            if body_line < 0.7:
                analysis['corrections'].append("Keep your body in a straight line")
                analysis['overall_score'] -= 1.5
            
            if 60 < angle < 110:
                analysis['feedback'].append("Perfect push-up depth!")
                analysis['overall_score'] += 1.0
            elif angle > 140:
                analysis['corrections'].append("Lower your chest more")
                analysis['overall_score'] -= 1.0
            
            hip_sag = hip.y - shoulder.y
            if hip_sag > 0.1:
                analysis['warnings'].append("⚠️ Hip sagging - engage your core")
                analysis['injury_risk'] = 'medium'
                
        except Exception as e:
            pass
        
        return analysis
    
    def _analyze_squat_form(self, landmarks, angle, analysis):
        try:
            hip = landmarks[23]
            knee = landmarks[25]
            ankle = landmarks[27]
            shoulder = landmarks[11]
            
            if angle < 90:
                analysis['feedback'].append("Excellent depth! Full range squat")
                analysis['overall_score'] += 1.5
            elif 90 < angle < 120:
                analysis['feedback'].append("Good squat, try going deeper")
                analysis['overall_score'] += 0.5
            elif angle > 150:
                analysis['corrections'].append("Squat down more, break parallel")
                analysis['overall_score'] -= 1.0
            
            back_lean = abs(shoulder.x - hip.x)
            if back_lean > 0.15:
                analysis['corrections'].append("Keep your chest up, back straight")
                analysis['overall_score'] -= 0.5
                
        except Exception as e:
            pass
        
        return analysis
    
    def _analyze_shoulder_press_form(self, landmarks, angle, analysis):
        try:
            shoulder = landmarks[11]
            elbow = landmarks[13]
            wrist = landmarks[15]
            
            if angle > 160:
                analysis['feedback'].append("Great overhead extension!")
                analysis['overall_score'] += 1.0
            elif angle < 90:
                analysis['corrections'].append("Press higher overhead")
                analysis['overall_score'] -= 1.0
            
            elbow_flare = abs(elbow.x - shoulder.x)
            if elbow_flare > 0.2:
                analysis['warnings'].append("⚠️ Elbows flaring - shoulder injury risk")
                analysis['injury_risk'] = 'high'
                
        except Exception as e:
            pass
        
        return analysis
    
    def _analyze_plank_form(self, landmarks, angle, analysis):
        try:
            shoulder = landmarks[11]
            hip = landmarks[23]
            ankle = landmarks[27]
            
            body_line_score = self._calculate_body_line([shoulder, hip, ankle])
            if body_line_score > 0.8:
                analysis['feedback'].append("Perfect plank alignment!")
                analysis['overall_score'] += 2.0
            elif body_line_score < 0.6:
                analysis['corrections'].append("Straighten your body line")
                analysis['overall_score'] -= 2.0
            
            hip_drop = hip.y - shoulder.y
            if hip_drop > 0.05:
                analysis['warnings'].append("⚠️ Hips sagging - core weakness")
                analysis['corrections'].append("Lift hips, engage core")
            elif hip_drop < -0.05:
                analysis['corrections'].append("Lower hips slightly")
                
        except Exception as e:
            pass
        
        return analysis
    
    def _calculate_body_line(self, points):
        if len(points) < 3:
            return 0.5
        
        try:
            total_deviation = 0
            for i in range(1, len(points) - 1):
                line_y = points[0].y + (points[-1].y - points[0].y) * (points[i].x - points[0].x) / (points[-1].x - points[0].x)
                deviation = abs(points[i].y - line_y)
                total_deviation += deviation
            
            score = max(0, 1.0 - total_deviation * 10)
            return score
        except:
            return 0.5
    
    def _default_form_score(self):
        return {
            'overall_score': 7.0,
            'feedback': ["Keep focusing on form"],
            'warnings': [],
            'corrections': [],
            'injury_risk': 'low'
        }

# INTELLIGENT FEEDBACK GENERATOR (NON-REPETITIVE)

class IntelligentFeedbackGenerator:
    """Advanced AI feedback system with context awareness and variety"""
    
    def __init__(self):
        self.model = model
        self.feedback_history = deque(maxlen=20)
        self.feedback_cache = {}
        self.ai_quota_exceeded = False
        self.context_memory = {
            'user_progress': 'beginner',
            'current_session_quality': 'good',
            'exercise_familiarity': {},
            'coaching_style': 'encouraging',
            'recent_achievements': [],
            'buddy_present': False
        }
        
        self.session_context = {
            'start_time': time.time(),
            'total_reps': 0,
            'exercise_switches': 0,
            'form_improvements': 0,
            'consistency_score': 8.0,
            'buddy_interactions': 0
        }
    
    def generate_contextual_feedback(self, feedback_type, exercise_name, **kwargs):
        """Generate intelligent, non-repetitive feedback"""
        
        self._update_session_context(feedback_type, exercise_name, **kwargs)
        context_key = self._create_context_key(feedback_type, exercise_name, kwargs)
        
        if self.ai_quota_exceeded:
            return self._get_intelligent_fallback(feedback_type, exercise_name, **kwargs)
        
        try:
            prompt = self._create_contextual_prompt(feedback_type, exercise_name, **kwargs)
            response = self.model.generate_content(prompt)
            
            if response.text:
                feedback = response.text.strip().replace('*', '').replace('"', '').replace('#', '')
                
                if not self._is_repetitive(feedback):
                    self.feedback_history.append(feedback)
                    self.feedback_cache[context_key] = feedback
                    print(f"🤖 AI: '{feedback}'")
                    return feedback
                else:
                    return self._get_feedback_variant(feedback_type, exercise_name, **kwargs)
            
        except Exception as e:
            if "quota" in str(e).lower():
                self.ai_quota_exceeded = True
                print("⚠️ AI quota exceeded, switching to intelligent fallbacks")
            else:
                print(f"❌ AI error: {e}")
        
        return self._get_intelligent_fallback(feedback_type, exercise_name, **kwargs)
    
    def _create_contextual_prompt(self, feedback_type, exercise_name, **kwargs):
        """Create intelligent prompts with full context"""
        
        session_minutes = (time.time() - self.session_context['start_time']) / 60
        recent_feedback = list(self.feedback_history)[-3:] if self.feedback_history else []
        
        base_context = f"""
        ADVANCED VOICEGYM CONTEXT:
        - User: {CURRENT_USER}
        - Exercise: {exercise_name}
        - Session time: {session_minutes:.1f} minutes
        - Total reps: {self.session_context['total_reps']}
        - User level: {self.context_memory['user_progress']}
        - Recent feedback: {recent_feedback}
        - Achievement tracking: Active
        - Social features: Enabled
        
        REQUIREMENTS:
        - Be completely different from recent feedback
        - Maximum 8 words
        - Motivational and specific
        - Easy to translate
        - Professional fitness coaching tone
        """
        
        if feedback_type == 'rep_completed':
            rep_count = kwargs.get('rep', 1)
            form_score = kwargs.get('form_score', 7.0)
            
            return base_context + f"""
            TYPE: Rep completion feedback for rep {rep_count}
            FORM QUALITY: {form_score}/10
            
            Generate unique congratulations that:
            - Celebrates this specific rep number
            - Acknowledges form quality
            - Varies from previous messages
            - Mentions achievement progress
            
            Examples:
            "Rep {rep_count}: Excellence tracked perfectly!"
            "Achievement rep {rep_count} - outstanding form!"
            "Progress rep {rep_count} - champion level!"
            """
        
        elif feedback_type == 'achievement_unlocked':
            achievement_name = kwargs.get('achievement_name', 'Achievement')
            return base_context + f"""
            TYPE: Achievement celebration
            ACHIEVEMENT: {achievement_name}
            
            Generate achievement celebration:
            - Celebrates specific achievement
            - Shows genuine excitement
            - Motivates continued progress
            
            Examples:
            "Achievement unlocked: {achievement_name}! Amazing!"
            "Milestone conquered: {achievement_name} earned!"
            "Victory achieved: {achievement_name} unlocked!"
            """
        
        elif feedback_type == 'global_challenge':
            challenge_name = kwargs.get('challenge_name', 'Global Challenge')
            contribution = kwargs.get('contribution', 0)
            return base_context + f"""
            TYPE: Global challenge participation
            CHALLENGE: {challenge_name}
            CONTRIBUTION: {contribution}
            
            Generate global challenge motivation:
            - Celebrates worldwide participation
            - Shows global impact
            - Motivates community involvement
            
            Examples:
            "Global challenge: {contribution} contributed worldwide!"
            "Community impact: helping {challenge_name}!"
            "Worldwide warrior: {contribution} reps logged!"
            """
        
        elif feedback_type == 'form_correction':
            correction = kwargs.get('correction', 'adjust form')
            return base_context + f"""
            TYPE: Form correction feedback
            SPECIFIC ISSUE: {correction}
            
            Generate helpful form cue:
            - Addresses specific issue
            - Encouraging not critical
            - Provides clear action
            - Builds confidence
            
            Examples:
            "Fine-tune technique: {correction}"
            "Small adjustment: {correction}"
            "Perfect with: {correction}"
            """
        
        elif feedback_type == 'streak_milestone':
            streak_days = kwargs.get('streak_days', 1)
            return base_context + f"""
            TYPE: Streak milestone celebration
            STREAK LENGTH: {streak_days} days
            
            Generate streak celebration:
            - Celebrates consistency
            - Acknowledges dedication
            - Motivates continuation
            
            Examples:
            "{streak_days} day streak! Consistency champion!"
            "Amazing {streak_days} day dedication!"
            "Unstoppable {streak_days} day streak!"
            """
        
        return base_context + "Generate appropriate motivational feedback."
    
    def _update_session_context(self, feedback_type, exercise_name, **kwargs):
        """Update session context"""
        if feedback_type == 'rep_completed':
            self.session_context['total_reps'] += 1
        
        # Track exercise familiarity
        if exercise_name not in self.context_memory['exercise_familiarity']:
            self.context_memory['exercise_familiarity'][exercise_name] = 1
        else:
            self.context_memory['exercise_familiarity'][exercise_name] += 1
        
        # Adjust user progress
        form_score = kwargs.get('form_score', 7.0)
        if form_score > 8.5:
            self.context_memory['user_progress'] = 'advanced'
        elif form_score > 6.5:
            self.context_memory['user_progress'] = 'intermediate'
        else:
            self.context_memory['user_progress'] = 'beginner'
    
    def _create_context_key(self, feedback_type, exercise_name, kwargs):
        key_data = f"{feedback_type}_{exercise_name}_{kwargs.get('rep', 0)}_{int(time.time()/10)}"
        return hashlib.md5(key_data.encode()).hexdigest()[:8]
    
    def _is_repetitive(self, feedback):
        if len(self.feedback_history) < 3:
            return False
        
        recent = list(self.feedback_history)[-3:]
        feedback_words = set(feedback.lower().split())
        
        for prev_feedback in recent:
            prev_words = set(prev_feedback.lower().split())
            overlap = len(feedback_words.intersection(prev_words))
            if overlap > len(feedback_words) * 0.6:
                return True
        
        return False
    
    def _get_feedback_variant(self, feedback_type, exercise_name, **kwargs):
        enhanced_variants = {
            'achievement_unlocked': [
                "Achievement unlocked! Incredible progress!",
                "New badge earned! Outstanding!",
                "Milestone reached! You're amazing!",
                "Achievement conquered! Fantastic!",
                "Goal smashed! Brilliant work!",
                "Badge unlocked! Exceptional effort!",
                "Milestone achieved! Inspiring!",
                "Achievement earned! Phenomenal!"
            ],
            'rep_completed': [
                "Rep mastery demonstrated beautifully!",
                "Perfect execution witnessed!",
                "Technique excellence achieved!",
                "Form mastery in action!",
                "Strength building successfully!",
                "Biomechanics beauty displayed!",
                "Power rep delivered!",
                "Champion rep completed!"
            ],
            'global_challenge': [
                "Global warrior contributing worldwide!",
                "Community champion in action!",
                "Worldwide impact maker!",
                "Global fitness ambassador!",
                "International challenge crusher!",
                "Community leader emerging!",
                "Global database synchronized!",
                "Worldwide progress tracked!"
            ],
            'form_correction': [
                "Fine-tune for perfection!",
                "Small tweak, big improvement!",
                "Optimize movement pattern!",
                "Enhance technique quality!",
                "Perfect with adjustment!",
                "Refine for excellence!",
                "Biomechanics optimization!",
                "Form enhancement activated!"
            ]
        }
        
        if feedback_type in enhanced_variants:
            return random.choice(enhanced_variants[feedback_type])
        
        return "Outstanding progress continues!"
    
    def _get_intelligent_fallback(self, feedback_type, exercise_name, **kwargs):
        """Enhanced fallback with social awareness"""
        
        rep_count = kwargs.get('rep', 1)
        form_score = kwargs.get('form_score', 7.0)
        
        if feedback_type == 'achievement_unlocked':
            achievement = kwargs.get('achievement_name', 'Achievement')
            return f"🏆 {achievement} unlocked! Amazing work!"
        
        elif feedback_type == 'global_challenge':
            return "Global fitness warrior contributing!"
        
        elif feedback_type == 'streak_milestone':
            days = kwargs.get('streak_days', 1)
            return f"{days} day streak! Consistency champion!"
        
        elif feedback_type == 'rep_completed':
            if form_score > 8.5:
                return f"Rep {rep_count}: Perfect form mastery!"
            elif form_score > 6.5:
                return f"Rep {rep_count}: Solid technique!"
            else:
                return f"Rep {rep_count}: Building strength!"
        
        elif feedback_type == 'form_correction':
            correction = kwargs.get('correction', 'adjust form')
            return f"Fine-tune: {correction}"
        
        return "Excellent progress continues!"
    
    def add_recent_achievement(self, achievement_name):
        """Add achievement to recent context"""
        self.context_memory['recent_achievements'].append(achievement_name)
        if len(self.context_memory['recent_achievements']) > 3:
            self.context_memory['recent_achievements'].pop(0)

# MURF SDK MANAGER

class AdvancedMurfSDKManager:
    def __init__(self, api_key):
        self.api_key = api_key
        self.translation_cache = {}
        self.audio_cache = {}
        
        try:
            from murf import Murf
            self.client = Murf(api_key=api_key)
            print("✅ Murf SDK initialized successfully!")
        except ImportError:
            print("📦 Installing Murf SDK...")
            os.system("pip install murf")
            try:
                from murf import Murf
                self.client = Murf(api_key=api_key)
                print("✅ Murf SDK installed and initialized!")
            except Exception as e:
                print(f"❌ Failed to install/import Murf SDK: {e}")
                self.client = None
        
        self._setup_direct_api()
        self.audio_dir = self._create_audio_directory()
        
        self.lang_mapping = {
            'hi-IN': 'hi-IN', 'zh-CN': 'zh-CN', 'fr-FR': 'fr-FR', 
            'de-DE': 'de-DE', 'es-ES': 'es-ES', 'en-US': 'en-US'
        }
    
    def _create_audio_directory(self):
        try:
            audio_dir = os.path.join(os.getcwd(), "voicegym_audio")
            os.makedirs(audio_dir, exist_ok=True)
            test_file = os.path.join(audio_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("VoiceGym test")
            os.remove(test_file)
            print(f"✅ Audio directory: {audio_dir}")
            return audio_dir
        except Exception as e:
            print(f"⚠️ Using temp directory for audio: {e}")
            return tempfile.gettempdir()
    
    def _setup_direct_api(self):
        self.base_url = "https://api.murf.ai/v1"
        self.headers = {"api-key": self.api_key, "Content-Type": "application/json"}
    
    def translate_with_murf_sdk(self, english_text, voice_config):
        target_lang_code = voice_config['lang_code']
        target_lang = self.lang_mapping.get(target_lang_code, 'en-US')
        
        if target_lang.startswith('en-'):
            return english_text
        
        cache_key = f"{english_text}_{target_lang}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        if self.client:
            try:
                response = self.client.text.translate(
                    target_language=target_lang,
                    texts=[english_text]
                )
                
                if response and hasattr(response, 'translations') and response.translations:
                    translated_text = response.translations[0].translated_text
                    if translated_text and translated_text.strip():
                        self.translation_cache[cache_key] = translated_text
                        return translated_text
            except Exception as e:
                print(f"❌ Murf SDK translation error: {e}")
        
        # Fallback to direct API
        try:
            payload = {"target_language": target_lang, "texts": [english_text]}
            response = requests.post(f"{self.base_url}/text/translate", 
                                   headers=self.headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if 'translations' in result and result['translations']:
                    translated = result['translations'][0].get('translated_text')
                    if translated:
                        self.translation_cache[cache_key] = translated
                        return translated
        except Exception as e:
            print(f"❌ Direct API error: {e}")
        
        return english_text
    
    def generate_speech_with_murf_tts(self, text, voice_config):
        voice_id = voice_config['voice_id']
        cache_key = f"{text}_{voice_id}"
        
        if cache_key in self.audio_cache:
            return self.audio_cache[cache_key]
        
        try:
            payload = {
                "text": text, "voiceId": voice_id, "format": "MP3",
                "model": "GEN2", "returnAsBase64": False,
                "language": voice_config['lang_code'], "speed": 1.0, "pitch": 0
            }
            
            response = requests.post(f"{self.base_url}/speech/generate", 
                                   headers=self.headers, json=payload, timeout=20)
            
            if response.status_code == 200:
                response_data = response.json()
                audio_url = response_data.get('audioFile')
                
                if audio_url:
                    audio_response = requests.get(audio_url, timeout=15)
                    if audio_response.status_code == 200:
                        audio_filename = os.path.join(
                            self.audio_dir,
                            f"voicegym_tts_{voice_id}_{int(time.time())}.mp3"
                        )
                        
                        with open(audio_filename, "wb") as f:
                            f.write(audio_response.content)
                        
                        self.audio_cache[cache_key] = audio_filename
                        print(f"✅ TTS generated: {os.path.basename(audio_filename)}")
                        return audio_filename
            
            print(f"❌ TTS failed: {response.status_code}")
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
        
        return None

# ADVANCED PIPELINE AUDIO SYSTEM

class AdvancedPipelineAudioSystem:
    def __init__(self, voice_config, feedback_generator, murf_manager):
        self.voice_config = voice_config
        self.feedback_generator = feedback_generator
        self.murf = murf_manager
        self.audio_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.playing = False
        self._shutdown = False
    
    def speak_advanced_pipeline(self, feedback_type, priority=False, **kwargs):
        if self._shutdown:
            return
            
        if priority:
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except:
                    break
        
        self.audio_queue.put((feedback_type, kwargs, priority))
        self.executor.submit(self._process_advanced_pipeline)
        return True
    
    def _process_advanced_pipeline(self):
        if self.playing or self._shutdown:
            return
            
        try:
            feedback_type, kwargs, priority = self.audio_queue.get_nowait()
            
            print(f"\n🎯 ADVANCED AI PIPELINE: {feedback_type}")
            print("=" * 70)
            
            # Generate intelligent feedback
            exercise_name = kwargs.get('exercise', 'Bicep Curls')
            english_text = self.feedback_generator.generate_contextual_feedback(
                feedback_type, exercise_name, **kwargs
            )
            print(f"✅ AI Generated: '{english_text}'")
            
            # Translate
            translated_text = self.murf.translate_with_murf_sdk(english_text, self.voice_config)
            print(f"✅ Translation: '{translated_text}'")
            
            # Generate speech
            audio_file = self.murf.generate_speech_with_murf_tts(translated_text, self.voice_config)
            
            # Play audio
            if audio_file:
                self._play_audio_file(audio_file)
                print(f"✅ PIPELINE COMPLETE!")
            
            print("=" * 70)
            
        except queue.Empty:
            pass
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
    
    def _play_audio_file(self, audio_filename):
        try:
            self.playing = True
            
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            pygame.mixer.music.load(audio_filename)
            pygame.mixer.music.play()
            
            timeout = time.time() + 30
            while pygame.mixer.music.get_busy() and time.time() < timeout and not self._shutdown:
                time.sleep(0.1)
                
        except Exception as e:
            print(f"❌ Audio playback error: {e}")
        finally:
            self.playing = False
    
    def shutdown(self):
        self._shutdown = True
        self.playing = False

# VIRTUAL WORKOUT BUDDY SYSTEM

class VirtualWorkoutBuddy:
    """AI workout companion that exercises alongside user"""
    
    def __init__(self, user_level='beginner'):
        self.user_level = user_level
        self.buddy_name = "Alex"
        self.buddy_reps = 0
        self.buddy_exercises = []
        self.buddy_motivation_style = "encouraging"
        self.buddy_stats = {
            'endurance': random.randint(7, 9),
            'form_quality': random.randint(8, 10),
            'consistency': random.randint(8, 9)
        }
    
    def start_buddy_workout(self, exercise_name):
        """Start working out with buddy"""
        self.buddy_reps = 0
        self.current_exercise = exercise_name
        
        rep_offset = random.choice([-1, 0, 1])
        self.buddy_reps = max(0, rep_offset)
        
        return f"{self.buddy_name} is ready! Let's do {exercise_name} together!"
    
    def buddy_rep_update(self, user_reps):
        """Update buddy's progress based on user"""
        if user_reps > self.buddy_reps:
            if random.random() < 0.7:
                self.buddy_reps = user_reps + random.choice([-1, 0])
        
        if user_reps > 0 and user_reps % 5 == 0:
            return self.get_buddy_motivation(user_reps)
        
        return None
    
    def get_buddy_motivation(self, user_reps):
        """Get motivational message from buddy"""
        if user_reps > self.buddy_reps:
            messages = [
                f"Wow! You're at {user_reps} reps, I'm at {self.buddy_reps}!",
                f"You're ahead of me! {user_reps} to {self.buddy_reps}!",
                f"I need to catch up! You're crushing it!"
            ]
        elif user_reps == self.buddy_reps:
            messages = [
                f"We're perfectly synced at {user_reps} reps!",
                f"Matching each other rep for rep!",
                f"Perfect teamwork at {user_reps}!"
            ]
        else:
            messages = [
                f"I'm at {self.buddy_reps}, you're at {user_reps} - great pace!",
                f"We're both doing amazing work!",
                f"Quality over quantity - your form is perfect!"
            ]
        
        return random.choice(messages)

# SOCIAL FEATURES MANAGER

class SocialFeaturesManager:
    """Manage family challenges and community features"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.init_global_challenges()
    
    def init_global_challenges(self):
        """Initialize global community challenges"""
        conn = self.db.get_connection()
        
        try:
            cursor = conn.cursor()
            
            # Check if we have active challenges
            cursor.execute("SELECT COUNT(*) FROM global_challenges WHERE is_active = TRUE")
            active_count = cursor.fetchone()[0]
            
            if active_count == 0:
                # Create default challenges
                challenges = [
                    {
                        'name': 'Global Push-up Challenge 2025',
                        'type': 'push_ups',
                        'target': 1000000,
                        'start_date': '2025-08-30',
                        'end_date': '2025-12-31'
                    },
                    {
                        'name': 'Million Rep September',
                        'type': 'total_reps',
                        'target': 1000000,
                        'start_date': '2025-08-30',
                        'end_date': '2025-09-30'
                    },
                    {
                        'name': 'Global Squat Quest',
                        'type': 'squats',
                        'target': 500000,
                        'start_date': '2025-08-30',
                        'end_date': '2025-10-15'
                    }
                ]
                
                for challenge in challenges:
                    cursor.execute('''
                        INSERT INTO global_challenges (challenge_name, challenge_type, target_value, current_value, start_date, end_date)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    ''', (
                        challenge['name'], challenge['type'], challenge['target'],
                        random.randint(50000, 200000),  # Simulate existing progress
                        challenge['start_date'], challenge['end_date']
                    ))
            
            conn.commit()
        except Exception as e:
            print(f"❌ Global challenges init error: {e}")
            conn.rollback()
        finally:
            self.db.return_connection(conn)

# HARDCODED VOICES DATABASE

def get_advanced_voices_database():
    return {
        "Hindi (India)": {
            "hi-IN": {
                "Male": [
                    {"id": "hi-IN-rahul", "name": "Rahul"},
                    {"id": "hi-IN-amit", "name": "Amit"},
                    {"id": "hi-IN-kabir", "name": "Kabir"}
                ],
                "Female": [
                    {"id": "hi-IN-shweta", "name": "Shweta"},
                    {"id": "hi-IN-ayushi", "name": "Ayushi"}
                ]
            }
        },
        "Chinese (Mandarin)": {
            "zh-CN": {
                "Male": [
                    {"id": "zh-CN-zhang", "name": "Zhang"},
                    {"id": "zh-CN-tao", "name": "Tao"}
                ],
                "Female": [
                    {"id": "zh-CN-jiao", "name": "Jiao"},
                    {"id": "zh-CN-wei", "name": "Wei"}
                ]
            }
        },
        "French (France)": {
            "fr-FR": {
                "Male": [
                    {"id": "fr-FR-maxime", "name": "Maxime"},
                    {"id": "fr-FR-louis", "name": "Louis"}
                ],
                "Female": [
                    {"id": "fr-FR-adélie", "name": "Adélie"},
                    {"id": "fr-FR-justine", "name": "Justine"}
                ]
            }
        },
        "English (US)": {
            "en-US": {
                "Male": [
                    {"id": "en-US-ken", "name": "Ken"},
                    {"id": "en-US-ryan", "name": "Ryan"}
                ],
                "Female": [
                    {"id": "en-US-natalie", "name": "Natalie"},
                    {"id": "en-US-samantha", "name": "Samantha"}
                ]
            }
        }
    }

# VOICE SELECTION

def select_advanced_voice():
    print("\n🎤 SELECT YOUR ADVANCED AI FITNESS COACH")
    print("🚀 Multi-Exercise Detection | 🎯 Form Analysis | 🧠 Intelligent Feedback")
    print("💾 Enterprise Database | 🌍 Global Challenges | 🏆 Achievement System")
    print("=" * 80)
    
    voices_db = get_advanced_voices_database()
    voice_options = []
    counter = 1
    
    for language, lang_codes in voices_db.items():
        print(f"\n🌍 {language}:")
        for lang_code, genders in lang_codes.items():
            for gender, voices in genders.items():
                print(f"  👤 {gender}:")
                for voice in voices:
                    print(f"    {counter}. {voice['name']} - Advanced AI Coach")
                    voice_options.append({
                        'language': language,
                        'lang_code': lang_code,
                        'voice_id': voice['id'],
                        'name': voice['name'],
                        'gender': gender
                    })
                    counter += 1
    
    print(f"\n💫 Choose your Advanced AI Coach (1-{len(voice_options)}): ", end="")
    
    while True:
        try:
            choice = int(input())
            if 1 <= choice <= len(voice_options):
                selected = voice_options[choice - 1]
                print(f"\n✅ Selected Advanced Coach: {selected['name']} ({selected['language']})")
                print(f"   🎭 Voice ID: {selected['voice_id']}")
                print(f"   🌍 Language: {selected['lang_code']}")
                print(f"   🚀 Features: Multi-Exercise, Form Analysis, Global Challenges")
                return selected
            else:
                print(f"❌ Enter 1-{len(voice_options)}: ", end="")
        except ValueError:
            print("❌ Enter a number: ", end="")


# MAIN ADVANCED GYM CLASS

class AdvancedMultiExerciseVoiceGym:
    def __init__(self, voice_config, audio_system, db_manager):
        self.voice_config = voice_config
        self.audio_system = audio_system
        self.db = db_manager
        
        # Get or create user with updated timestamp
        self.user_id = self.db.get_or_create_user(CURRENT_USER)
        print(f"✅ User ID: {self.user_id}")
        
        # Initialize systems
        self.exercise_detector = SmartExerciseDetector()
        self.form_analyzer = AdvancedFormAnalyzer()
        self.buddy = VirtualWorkoutBuddy()
        
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Core tracking
        self.reps = 0
        self.stage = "ready"
        self.last_feedback = 0
        self.last_rep = 0
        self.last_speech = 0
        self.start_time = time.time()
        self.current_exercise = 'bicep_curls'
        self.last_exercise_change = 0
        
        # Form tracking
        self.current_form_score = 7.0
        self.form_feedback_timer = 0
        
        # Exercise data
        self.exercises = ExerciseDefinitions.get_exercises()
        self.exercise_data = self.exercises[self.current_exercise]
        
        # Session tracking for database
        self.session_id = str(uuid.uuid4())
        self.workout_data = {
            'duration': 0,
            'total_reps': 0,
            'exercises': {},
            'avg_form_score': 7.0,
            'perfect_reps': 0,
            'exercise_switches': 0,
            'ai_calls': 0,
            'form_corrections': 0,
            'injury_warnings': 0,
            'voice_coach': voice_config['name'],
            'language': voice_config['language']
        }
        
        # Stats tracking
        self.advanced_stats = {
            'total_exercises_detected': 1,
            'form_corrections_given': 0,
            'injury_warnings': 0,
            'perfect_reps': 0,
            'exercise_switches': 0,
            'ai_feedback_calls': 0,
            'global_contributions': 0
        }
        
        self.is_paused = False
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            raise Exception("Cannot open camera!")
        
        print("✅ Advanced Multi-Exercise VoiceGym initialized!")
        print("🎯 Features: Smart Detection, Form Analysis, Database Tracking, Global Challenges")
    
    def calculate_angle(self, landmarks):
        """Calculate angle for current exercise"""
        try:
            indices = self.exercise_data['landmarks']
            if len(indices) >= 3:
                a = np.array([landmarks[indices[0]].x, landmarks[indices[0]].y])
                b = np.array([landmarks[indices[1]].x, landmarks[indices[1]].y])
                c = np.array([landmarks[indices[2]].x, landmarks[indices[2]].y])
                
                radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
                angle = np.abs(radians * 180.0 / np.pi)
                return 360 - angle if angle > 180 else angle
        except:
            pass
        return 90
    
    def speak_advanced(self, feedback_type, priority=False, **kwargs):
        """Advanced speech with intelligent feedback"""
        kwargs['exercise'] = self.exercise_data['name']
        kwargs['form_score'] = self.current_form_score
        
        self.audio_system.speak_advanced_pipeline(feedback_type, priority, **kwargs)
        self.advanced_stats['ai_feedback_calls'] += 1
        self.workout_data['ai_calls'] += 1
    
    def process_frame(self, frame):
        """Advanced frame processing with multi-exercise detection"""
        current_time = time.time()
        
        # Update workout duration
        self.workout_data['duration'] = (current_time - self.start_time) / 60
        
        if self.is_paused:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.putText(frame, 'PAUSED - Press P to resume', (w//2 - 200, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            return frame
        
        # Pose detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        angle = 90
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Smart exercise detection
            detected_exercise = self.exercise_detector.detect_exercise(landmarks)
            
            # Handle exercise change
            if detected_exercise != self.current_exercise and current_time - self.last_exercise_change > 5:
                print(f"🔄 Exercise change: {self.current_exercise} → {detected_exercise}")
                self.current_exercise = detected_exercise
                self.exercise_data = self.exercises[detected_exercise]
                self.last_exercise_change = current_time
                self.advanced_stats['exercise_switches'] += 1
                self.workout_data['exercise_switches'] += 1
                
                # Announce exercise change
                self.speak_advanced('exercise_switch', priority=True, 
                                  new_exercise=self.exercise_data['name'])
            
            # Calculate angle for current exercise
            angle = self.calculate_angle(landmarks)
            
            # Advanced form analysis
            form_analysis = self.form_analyzer.analyze_form(landmarks, self.current_exercise, angle)
            self.current_form_score = form_analysis['overall_score']
            
            # Update workout data for current exercise
            if self.current_exercise not in self.workout_data['exercises']:
                self.workout_data['exercises'][self.current_exercise] = {
                    'reps': 0,
                    'best_form': 0.0
                }
            
            # Rep counting with exercise-specific logic
            if angle > self.exercise_data['down_threshold'] and self.stage != "down":
                self.stage = "down"
            elif (angle < self.exercise_data['up_threshold'] and 
                  self.stage == "down" and 
                  current_time - self.last_rep > 2.0):
                self.stage = "up"
                self.reps += 1
                self.last_rep = current_time
                
                # Update exercise stats
                self.workout_data['exercises'][self.current_exercise]['reps'] = self.reps
                self.workout_data['exercises'][self.current_exercise]['best_form'] = max(
                    self.workout_data['exercises'][self.current_exercise]['best_form'],
                    form_analysis['overall_score']
                )
                self.workout_data['total_reps'] = self.reps
                
                # Track perfect reps
                if form_analysis['overall_score'] > 8.5:
                    self.advanced_stats['perfect_reps'] += 1
                    self.workout_data['perfect_reps'] += 1
                
                # Rep completion with form score
                if current_time - self.last_speech > 4:
                    self.speak_advanced('rep_completed', 
                                      rep=self.reps, 
                                      form_score=form_analysis['overall_score'])
                    self.last_speech = current_time
                    print(f"✅ Rep {self.reps} completed! Form: {form_analysis['overall_score']:.1f}/10")
            
            # Form correction feedback
            if (form_analysis['corrections'] and 
                current_time - self.form_feedback_timer > 8 and
                current_time - self.last_speech > 6):
                
                correction = form_analysis['corrections'][0]
                self.speak_advanced('form_correction', 
                                  correction=correction,
                                  form_score=form_analysis['overall_score'])
                self.form_feedback_timer = current_time
                self.last_speech = current_time
                self.advanced_stats['form_corrections_given'] += 1
                self.workout_data['form_corrections'] += 1
            
            # Injury warnings
            if (form_analysis['warnings'] and form_analysis['injury_risk'] in ['medium', 'high']):
                if current_time - self.last_speech > 3:
                    warning = form_analysis['warnings'][0]
                    self.speak_advanced('safety_warning', priority=True, warning=warning)
                    self.last_speech = current_time
                    self.advanced_stats['injury_warnings'] += 1
                    self.workout_data['injury_warnings'] += 1
            
            # General coaching
            if (current_time - self.last_feedback > 20 and 
                current_time - self.last_speech > 8):
                
                self.speak_advanced('coaching',
                                  quality='excellent' if form_analysis['overall_score'] > 7 else 'good')
                self.last_feedback = current_time
                self.last_speech = current_time
            
            # Enhanced pose drawing with form colors
            self._draw_enhanced_pose(frame, results, form_analysis)
        
        return self.draw_advanced_overlay(frame, angle)
    
    def _draw_enhanced_pose(self, frame, results, form_analysis):
        """Draw pose with form-based coloring"""
        
        # Choose colors based on form quality
        if form_analysis['overall_score'] > 8.5:
            landmark_color = (0, 255, 0)      # Green for excellent
            connection_color = (0, 255, 0)
        elif form_analysis['overall_score'] > 6.5:
            landmark_color = (0, 255, 255)    # Yellow for good
            connection_color = (0, 255, 255)
        else:
            landmark_color = (0, 165, 255)    # Orange for needs improvement
            connection_color = (0, 165, 255)
        
        # Draw pose landmarks with form-based colors
        self.mp_draw.draw_landmarks(
            frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=landmark_color, thickness=3, circle_radius=4),
            self.mp_draw.DrawingSpec(color=connection_color, thickness=2)
        )
    
    def draw_advanced_overlay(self, frame, angle):
        """Advanced overlay with multi-exercise info"""
        h, w = frame.shape[:2]
        
        # Main overlay
        cv2.rectangle(frame, (10, 10), (min(950, w-10), 200), (0,0,0), -1)
        
        # Title
        cv2.putText(frame, f'🚀 ULTIMATE ADVANCED VOICEGYM - Enterprise Edition', 
                   (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        # User and session info
        cv2.putText(frame, f'👤 User: {CURRENT_USER} | 💾 Session: {self.session_id[:8]}... | 🎤 {self.voice_config["name"]}', 
                   (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,255,255), 2)
        
        # Current exercise and stats
        cv2.putText(frame, f'🏋️ Exercise: {self.exercise_data["name"]} | Reps: {self.reps} | Angle: {angle:.0f}° | Stage: {self.stage} | Form: {self.current_form_score:.1f}/10', 
                   (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
        
        # Session stats
        cv2.putText(frame, f'📊 Duration: {self.workout_data["duration"]:.1f}min | Perfect Reps: {self.advanced_stats["perfect_reps"]} | Switches: {self.advanced_stats["exercise_switches"]}', 
                   (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 2)
        
        # Advanced features
        cv2.putText(frame, f'🤖 AI Calls: {self.advanced_stats["ai_feedback_calls"]} | Form Corrections: {self.advanced_stats["form_corrections_given"]} | Safety Warnings: {self.advanced_stats["injury_warnings"]}', 
                   (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 150, 255), 2)
        
        # Database and features info
        cv2.putText(frame, f'💾 Enterprise Database | 🌍 Global Challenges | 🏆 Achievements | 🎯 Multi-Exercise AI | 📊 Progress Tracking', 
                   (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 255), 1)
        
        # Controls
        cv2.putText(frame, 'Controls: P=Pause | Q=Quit | S=Save | R=Stats | 1-6=Manual Exercise Switch | Auto-detection Active', 
                   (15, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 255), 1)
        
        return frame
    
    def save_current_workout(self):
        """Save current workout to database"""
        if self.reps > 0:
            try:
                # Calculate average form score
                total_form = sum(self.workout_data['exercises'][ex]['best_form'] 
                               for ex in self.workout_data['exercises'])
                self.workout_data['avg_form_score'] = total_form / len(self.workout_data['exercises']) if self.workout_data['exercises'] else 7.0
                
                # Save to database
                workout_id, new_achievements = self.db.save_workout(self.user_id, self.workout_data)
                
                # Contribute to global challenges
                contributions = self.db.contribute_to_global_challenges(self.user_id, self.workout_data)
                self.advanced_stats['global_contributions'] += len(contributions)
                
                print(f"✅ Workout saved! ID: {workout_id}")
                
                if new_achievements:
                    print(f"🏆 New achievements: {', '.join(new_achievements)}")
                    for achievement in new_achievements:
                        self.speak_advanced('achievement_unlocked', priority=True, 
                                          achievement_name=achievement)
                        self.audio_system.feedback_generator.add_recent_achievement(achievement)
                
                if contributions:
                    print(f"🌍 Global contributions: {len(contributions)}")
                    for contrib in contributions:
                        self.speak_advanced('global_challenge', priority=False,
                                          challenge_name=contrib['challenge'],
                                          contribution=contrib['contribution'])
                
            except Exception as e:
                print(f"❌ Save error: {e}")
    
    def show_realtime_stats(self):
        """Show real-time statistics"""
        try:
            stats = self.db.get_user_stats(self.user_id)
            leaderboard = self.db.get_global_leaderboard(limit=5)
            
            print("\n" + "="*100)
            print("📊 ULTIMATE VOICEGYM REAL-TIME STATISTICS")
            print("="*100)
            
            if stats['user']:
                user = stats['user']
                print(f"👤 User: {user['username']}")
                print(f"🏋️ Total Workouts: {user['total_workouts']}")
                print(f"🔢 Total Reps: {user['total_reps']}")
                print(f"🔥 Current Streak: {user['current_streak']} days")
                print(f"🏆 Longest Streak: {user['longest_streak']} days")
                print(f"⏱️ Total Time: {float(user['total_workout_minutes']):.1f} minutes")
                print(f"🌍 Global Rank: #{stats['global_rank']}")
            
            print(f"\n🏆 TOP 5 GLOBAL LEADERBOARD:")
            for i, leader in enumerate(leaderboard[:5], 1):
                print(f"   {i}. {leader['username']}: {leader['total_reps']} reps")
            
            print(f"\n🎖️ RECENT ACHIEVEMENTS:")
            for achievement in stats['achievements'][:3]:
                print(f"   🏆 {achievement['achievement_name']}: {achievement['description']}")
            
            print(f"\n📈 CURRENT SESSION:")
            print(f"   🏋️ Exercise: {self.exercise_data['name']}")
            print(f"   🔢 Reps: {self.reps}")
            print(f"   🎯 Form Score: {self.current_form_score:.1f}/10")
            print(f"   ⏱️ Duration: {self.workout_data['duration']:.1f} minutes")
            print(f"   🤖 AI Feedback Calls: {self.advanced_stats['ai_feedback_calls']}")
            
            print("="*100)
            
        except Exception as e:
            print(f"❌ Stats display error: {e}")
    
    def run_advanced_workout(self):
        """Main advanced workout loop"""
        print("🎥 Starting ULTIMATE ADVANCED VOICEGYM...")
        print(f"🎤 Voice: {self.voice_config['name']} ({self.voice_config['language']})")
        print(f"💾 Database: Enterprise PostgreSQL")
        print(f"🆔 Session: {self.session_id}")
        print(f"🚀 Features: 8 Exercises, Form Analysis, Global Challenges, Achievement System")
        print("⌨️ Controls: P=Pause, Q=Quit, S=Save, R=Stats, 1-6=Manual Exercise Switch")
        print("=" * 120)
        
        # Get user stats at start
        user_stats = self.db.get_user_stats(self.user_id)
        if user_stats['user']:
            print(f"📊 Welcome back! Total workouts: {user_stats['user']['total_workouts']}")
            print(f"   Total reps: {user_stats['user']['total_reps']} | Current streak: {user_stats['user']['current_streak']}")
            print(f"   Global rank: #{user_stats['global_rank']}")
        
        # Update streak
        current_streak = self.db.update_streak(self.user_id)
        print(f"🔥 Current workout streak: {current_streak} days!")
        
        if current_streak >= 7:
            self.speak_advanced('streak_milestone', priority=True, streak_days=current_streak)
        
        # Advanced welcome message
        self.speak_advanced('welcome', priority=True, exercise=self.exercise_data['name'])
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            processed_frame = self.process_frame(frame)
            
            cv2.imshow('🚀 Ultimate Advanced VoiceGym - Enterprise Edition', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('p'):
                self.is_paused = not self.is_paused
                if self.is_paused:
                    self.speak_advanced('paused', priority=True)
                else:
                    self.speak_advanced('resume', priority=True)
            elif key == ord('s'):  # Manual save
                self.save_current_workout()
            elif key == ord('r'):  # Show stats
                self.show_realtime_stats()
            elif key == ord('1'):  # Manual exercise switches
                self.current_exercise = 'bicep_curls'
                self.exercise_data = self.exercises[self.current_exercise]
                self.speak_advanced('exercise_switch', priority=True, new_exercise=self.exercise_data['name'])
            elif key == ord('2'):
                self.current_exercise = 'push_ups'
                self.exercise_data = self.exercises[self.current_exercise]
                self.speak_advanced('exercise_switch', priority=True, new_exercise=self.exercise_data['name'])
            elif key == ord('3'):
                self.current_exercise = 'squats'
                self.exercise_data = self.exercises[self.current_exercise]
                self.speak_advanced('exercise_switch', priority=True, new_exercise=self.exercise_data['name'])
            elif key == ord('4'):
                self.current_exercise = 'shoulder_press'
                self.exercise_data = self.exercises[self.current_exercise]
                self.speak_advanced('exercise_switch', priority=True, new_exercise=self.exercise_data['name'])
            elif key == ord('5'):
                self.current_exercise = 'plank'
                self.exercise_data = self.exercises[self.current_exercise]
                self.speak_advanced('exercise_switch', priority=True, new_exercise=self.exercise_data['name'])
            elif key == ord('6'):
                self.current_exercise = 'lunges'
                self.exercise_data = self.exercises[self.current_exercise]
                self.speak_advanced('exercise_switch', priority=True, new_exercise=self.exercise_data['name'])
            
            frame_count += 1
            
            # Auto-save every 5 minutes
            if frame_count % 9000 == 0:  # Approximately 5 minutes at 30fps
                self.save_current_workout()
                print(f"📊 Auto-save: Frame {frame_count} | Reps: {self.reps} | Duration: {self.workout_data['duration']:.1f}min")
        
        # Final save and cleanup
        self.finalize_workout()
    
    def finalize_workout(self):
        """Finalize and save complete workout session"""
        try:
            # Final save
            self.save_current_workout()
            
            # Generate progress chart
            chart_path = self.db.generate_progress_chart(self.user_id)
            
            # Get final stats
            final_stats = self.db.get_user_stats(self.user_id)
            
            # Cleanup
            self.audio_system.shutdown()
            self.cap.release()
            cv2.destroyAllWindows()
            
            # Display comprehensive summary
            elapsed = (time.time() - self.start_time) / 60
            
            print("\n" + "="*120)
            print("🏁 ULTIMATE ADVANCED VOICEGYM WORKOUT COMPLETE!")
            print("="*120)
            print(f"👤 User: {CURRENT_USER}")
            print(f"🆔 Session ID: {self.session_id}")
            print(f"⏱️ Duration: {elapsed:.1f} minutes")
            print(f"🔢 Total Reps: {self.reps}")
            print(f"🎯 Form Score: {self.current_form_score:.1f}/10")
            print(f"🏋️ Current Exercise: {self.exercise_data['name']}")
            print(f"🤖 Voice Coach: {self.voice_config['name']} ({self.voice_config['language']})")
            
            if final_stats['user']:
                user = final_stats['user']
                print(f"\n📊 LIFETIME STATISTICS:")
                print(f"   🏋️ Total Workouts: {user['total_workouts']}")
                print(f"   🔢 Total Reps: {user['total_reps']}")
                print(f"   🔥 Current Streak: {user['current_streak']} days")
                print(f"   🏆 Longest Streak: {user['longest_streak']} days")
                print(f"   ⏱️ Total Time: {float(user['total_workout_minutes']):.1f} minutes")
                print(f"   🌍 Global Rank: #{final_stats['global_rank']}")
            
            print(f"\n🎯 SESSION ACHIEVEMENTS:")
            print(f"   ⭐ Perfect Form Reps: {self.advanced_stats['perfect_reps']}")
            print(f"   🔄 Exercise Switches: {self.advanced_stats['exercise_switches']}")
            print(f"   🤖 AI Feedback Calls: {self.advanced_stats['ai_feedback_calls']}")
            print(f"   🔧 Form Corrections: {self.advanced_stats['form_corrections_given']}")
            print(f"   ⚠️ Safety Warnings: {self.advanced_stats['injury_warnings']}")
            print(f"   🌍 Global Contributions: {self.advanced_stats['global_contributions']}")
            
            if chart_path:
                print(f"📈 Progress chart saved: {chart_path}")
            
            print("\n🚀 ENTERPRISE FEATURES DEMONSTRATED:")
            print("   ✅ PostgreSQL Enterprise Database with UUID Keys")
            print("   ✅ JSONB Storage for Flexible Exercise Data")
            print("   ✅ Multi-Exercise AI Detection (8+ exercises)")
            print("   ✅ Advanced Form Analysis with Injury Prevention")
            print("   ✅ Intelligent Non-Repetitive AI Feedback")
            print("   ✅ Real-time Global Challenge Participation")
            print("   ✅ Comprehensive Achievement System")
            print("   ✅ Murf SDK Multilingual TTS Integration")
            print("   ✅ Advanced Progress Tracking & Analytics")
            print("   ✅ Enterprise-Grade Performance & Scalability")
            
            print("\n🎖️ WORKOUT ACHIEVEMENTS:")
            if self.advanced_stats['perfect_reps'] > 5:
                print(f"   🏆 FORM MASTER: {self.advanced_stats['perfect_reps']} perfect reps!")
            if self.advanced_stats['exercise_switches'] > 2:
                print(f"   🔄 MULTI-EXERCISE CHAMPION: {self.advanced_stats['exercise_switches']} exercise switches!")
            if elapsed > 5:
                print(f"   ⏱️ ENDURANCE WARRIOR: {elapsed:.1f} minute workout!")
            if self.advanced_stats['form_corrections_given'] == 0:
                print(f"   ✨ PERFECT TECHNIQUE: No form corrections needed!")
            if self.advanced_stats['global_contributions'] > 0:
                print(f"   🌍 GLOBAL CONTRIBUTOR: {self.advanced_stats['global_contributions']} challenge contributions!")
            
            print("="*120)
            print(f"🎯 Built by {CURRENT_USER} with cutting-edge AI technology!")
            print(f"📅 Session completed: {CURRENT_DATE} {CURRENT_TIME} UTC")
            print("="*120)
            
        except Exception as e:
            print(f"❌ Workout finalization error: {e}")
        finally:
            # Close database connections
            self.db.close_pool()

# MAIN EXECUTION WITH UPDATED TIMESTAMP

if __name__ == "__main__":
    try:
        print("🚀 ULTIMATE ADVANCED VOICEGYM LAUNCHING...")
        print(f"📅 Session Date: {CURRENT_DATE} {CURRENT_TIME} UTC")
        print(f"👤 User: {CURRENT_USER}")
        print("🎯 Revolutionary Features:")
        print("   ✅ 8+ Exercise Smart Detection (Bicep Curls, Push-ups, Squats, Lunges, etc.)")
        print("   ✅ Real-time Form Analysis & Injury Prevention")
        print("   ✅ Intelligent Non-Repetitive AI Feedback")
        print("   ✅ Enterprise PostgreSQL Database")
        print("   ✅ Global Challenges & Social Features")
        print("   ✅ Comprehensive Achievement System")
        print("   ✅ Murf SDK Multilingual TTS")
        print("   ✅ Advanced Progress Tracking")
        print("=" * 120)
        
        # Check database connection requirements
        required_vars = ['POSTGRES_HOST', 'POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"❌ Missing database environment variables: {', '.join(missing_vars)}")
            print("📝 Please add to your .env file:")
            for var in missing_vars:
                if var == 'POSTGRES_PASSWORD':
                    print(f"   {var}=your_secure_password_here")
                elif var == 'POSTGRES_DB':
                    print(f"   {var}=voicegym")
                elif var == 'POSTGRES_USER':
                    print(f"   {var}=postgres")
                else:
                    print(f"   {var}=localhost")
            raise SystemExit()
        
        # Initialize database
        print("\n💾 Initializing enterprise database...")
        db_manager = ProgressDatabaseManager()
        
        # Initialize social features
        print("🌍 Setting up global challenges...")
        social_manager = SocialFeaturesManager(db_manager)
        
        # Select voice
        voice_config = select_advanced_voice()
        
        if not voice_config:
            print("❌ Voice configuration failed")
            exit(1)
        
        # Initialize AI systems
        print("\n🔧 Initializing advanced AI systems...")
        print("   🤖 Loading Intelligent Feedback Generator...")
        feedback_generator = IntelligentFeedbackGenerator()
        
        print("   🌍 Initializing Murf SDK Manager...")
        murf_manager = AdvancedMurfSDKManager(MURF_API_KEY)
        
        print("   🎵 Setting up Advanced Audio Pipeline...")
        audio_system = AdvancedPipelineAudioSystem(voice_config, feedback_generator, murf_manager)
        
        print("✅ All advanced systems initialized!")
        
        # Create advanced gym
        print("🏋️ Creating Ultimate Advanced VoiceGym...")
        gym = AdvancedMultiExerciseVoiceGym(voice_config, audio_system, db_manager)
        
        print("\n🎮 CONTROLS:")
        print("   P = Pause/Resume")
        print("   Q = Quit")
        print("   S = Save Workout to Database")
        print("   R = Show Real-time Statistics")
        print("   1 = Switch to Bicep Curls")
        print("   2 = Switch to Push-ups") 
        print("   3 = Switch to Squats")
        print("   4 = Switch to Shoulder Press")
        print("   5 = Switch to Plank")
        print("   6 = Switch to Lunges")
        print("   (Auto-detection also works!)")
        
        # Start advanced workout
        gym.run_advanced_workout()
        
    except KeyboardInterrupt:
        print("\n👋 Advanced workout interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            cv2.destroyAllWindows()
            pygame.mixer.quit()
        except:
            pass
        print("🏁 Ultimate Advanced VoiceGym session ended!")
        print("🎯 Thank you for using the most advanced AI fitness coach!")
        print(f"💪 Built by {CURRENT_USER} with enterprise-grade technology!")
        print(f"📅 Session ended: {CURRENT_DATE} {CURRENT_TIME} UTC")
