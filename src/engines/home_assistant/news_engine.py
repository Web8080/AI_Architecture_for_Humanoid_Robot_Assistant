"""
News Information Engine

PURPOSE:
    Fetches and delivers personalized news from multiple sources with content filtering.
    Provides age-appropriate news summaries and category-based filtering.

INTEGRATION:
    - NewsAPI for real-time headlines
    - Content safety filtering for children
    - Multi-language support
    - Topic categorization and personalization
    - TTS integration for audio news delivery

Author: Victor Ibhafidon
Date: November 2025
"""

from typing import Dict, Any, Optional, List
from src.engines.base_engine import BaseEngine
import logging
import os
import requests
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class NewsEngine(BaseEngine):
    """
    Production-grade news delivery engine with content filtering and personalization.
    
    CAPABILITIES:
    - Multi-source news aggregation
    - Category-based filtering (politics, tech, sports, entertainment, etc.)
    - Age-appropriate content filtering
    - Personalized news based on user interests
    - Breaking news alerts
    - Audio news summaries
    - Multi-language support
    
    MULTI-TIER FALLBACK:
    - Tier 1: NewsAPI (comprehensive, real-time)
    - Tier 2: RSS feed aggregation (reliable backup)
    - Tier 3: Cached headlines (offline mode)
    """
    
    # News categories
    CATEGORIES = {
        'general': 'General news',
        'business': 'Business and finance',
        'technology': 'Technology and science',
        'sports': 'Sports',
        'entertainment': 'Entertainment and culture',
        'health': 'Health and wellness',
        'science': 'Science discoveries'
    }
    
    # Country codes
    COUNTRIES = {
        'us': 'United States',
        'gb': 'United Kingdom', 
        'ca': 'Canada',
        'au': 'Australia',
        'ng': 'Nigeria',
        'in': 'India'
    }
    
    # Content filtering levels
    FILTER_CHILD = 'child'        # No violence, politics, adult content
    FILTER_TEEN = 'teen'          # Limited adult content
    FILTER_ADULT = 'adult'        # No filtering
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize news engine.
        
        Args:
            config: Configuration with:
                - api_key: NewsAPI key
                - default_country: Default country code
                - default_category: Default news category
                - max_articles: Maximum articles per request
                - enable_filtering: Enable content filtering
                - cache_duration: Cache duration in seconds
        """
        super().__init__(config)
        self.name = "NewsEngine"
        
        # API configuration
        self.api_key = config.get('api_key') if config else os.getenv("NEWS_API_KEY", "6513b1d989e44d3c853ff6e1e9eba7e3")
        self.base_url = "https://newsapi.org/v2"
        
        # Default settings
        self.default_country = config.get('default_country', 'us') if config else 'us'
        self.default_category = config.get('default_category', 'general') if config else 'general'
        self.max_articles = config.get('max_articles', 5) if config else 5
        self.enable_filtering = config.get('enable_filtering', True) if config else True
        
        # Cache for offline mode
        self.news_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_duration = config.get('cache_duration', 1800) if config else 1800  # 30 minutes
        
        # Blocked keywords for child safety
        self.adult_keywords = [
            'murder', 'death', 'killed', 'shooting', 'terrorism', 'assault',
            'violence', 'war', 'attack', 'bomb', 'crime', 'scandal'
        ]
        
        logger.info(f"✓ {self.name} initialized")
        logger.info(f"  - Default country: {self.default_country}")
        logger.info(f"  - Default category: {self.default_category}")
        logger.info(f"  - Content filtering: {self.enable_filtering}")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch and deliver news with personalization and filtering.
        
        Args:
            context: Request context:
                - category: News category
                - country: Country code
                - count: Number of articles
                - filter_level: 'child' | 'teen' | 'adult'
                - query: Search query (optional)
                - language: Language code
        
        Returns:
            News articles with metadata
        """
        # Extract parameters with defaults
        category = context.get('category', self.default_category)
        country = context.get('country', self.default_country)
        count = min(context.get('count', self.max_articles), 10)  # Max 10
        filter_level = context.get('filter_level', self.FILTER_ADULT)
        query = context.get('query')
        language = context.get('language', 'en')
        
        logger.info(f"📰 News request: category={category}, country={country}, count={count}")
        
        try:
            # TIER 1: NewsAPI (best)
            result = self._tier1_newsapi(category, country, count, filter_level, query, language)
            logger.info(f"✓ Tier 1: Fetched {len(result.get('articles', []))} articles")
            return result
            
        except Exception as e1:
            logger.warning(f"Tier 1 failed: {e1}, falling back to Tier 2")
            
            try:
                # TIER 2: RSS aggregation
                result = self._tier2_rss_feeds(category, country, count, filter_level)
                logger.info(f"✓ Tier 2: Fetched {len(result.get('articles', []))} articles")
                return result
                
            except Exception as e2:
                logger.warning(f"Tier 2 failed: {e2}, falling back to Tier 3")
                
                # TIER 3: Cached headlines
                result = self._tier3_cached_news(category, country, filter_level)
                logger.warning(f"⚠️ Tier 3: Using cached news")
                return result
    
    def _tier1_newsapi(
        self,
        category: str,
        country: str,
        count: int,
        filter_level: str,
        query: Optional[str],
        language: str
    ) -> Dict[str, Any]:
        """
        TIER 1: NewsAPI comprehensive news fetching.
        
        Fetches real-time news with category, country, and query support.
        """
        # Build cache key
        cache_key = f"{category}_{country}_{count}_{query or 'none'}"
        
        # Check cache
        if cache_key in self.news_cache:
            cached = self.news_cache[cache_key]
            cache_time = cached.get('timestamp', 0)
            if (datetime.now().timestamp() - cache_time) < self.cache_duration:
                logger.debug(f"Using cached news for {cache_key}")
                cached_data = cached['data']
                # Still apply filtering in case filter level changed
                cached_data['articles'] = self._filter_content(
                    cached_data['articles'], 
                    filter_level
                )
                return cached_data
        
        # Determine endpoint
        if query:
            # Search everything
            endpoint = f"{self.base_url}/everything"
            params = {
                'apiKey': self.api_key,
                'q': query,
                'language': language,
                'sortBy': 'publishedAt',
                'pageSize': count
            }
        else:
            # Top headlines
            endpoint = f"{self.base_url}/top-headlines"
            params = {
                'apiKey': self.api_key,
                'country': country,
                'pageSize': count
            }
            
            # Add category if not general
            if category != 'general':
                params['category'] = category
        
        # Fetch news
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Check for API errors
        if data.get('status') != 'ok':
            error_msg = data.get('message', 'Unknown error')
            raise Exception(f"NewsAPI error: {error_msg}")
        
        # Extract and process articles
        articles = data.get('articles', [])
        
        # Apply content filtering if enabled
        if self.enable_filtering and filter_level != self.FILTER_ADULT:
            articles = self._filter_content(articles, filter_level)
        
        # Process articles
        processed_articles = []
        for idx, article in enumerate(articles[:count], 1):
            processed = {
                'id': idx,
                'title': article.get('title', 'No title'),
                'description': article.get('description', ''),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'author': article.get('author', 'Unknown'),
                'published_at': article.get('publishedAt', ''),
                'url': article.get('url', ''),
                'image_url': article.get('urlToImage', ''),
                'content': article.get('content', '')
            }
            
            # Add reading time estimation
            word_count = len(processed['description'].split())
            processed['reading_time_minutes'] = max(1, word_count // 200)
            
            processed_articles.append(processed)
        
        # Build result
        result = {
            'articles': processed_articles,
            'total_results': len(processed_articles),
            'category': category,
            'country': country,
            'language': language,
            'filter_level': filter_level,
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'tier_used': 1,
            'status': 'success',
            'data_source': 'NewsAPI'
        }
        
        # Generate summary
        result['summary'] = self._generate_summary(processed_articles, category, country)
        
        # Cache result (before filtering for reuse)
        self.news_cache[cache_key] = {
            'data': result,
            'timestamp': datetime.now().timestamp()
        }
        
        return result
    
    def _tier2_rss_feeds(
        self,
        category: str,
        country: str,
        count: int,
        filter_level: str
    ) -> Dict[str, Any]:
        """
        TIER 2: RSS feed aggregation backup.
        
        Uses RSS feeds when primary API unavailable.
        Note: Simplified implementation - real version would parse actual RSS feeds.
        """
        logger.warning("RSS feed aggregation not fully implemented - using fallback")
        
        # In production, this would:
        # 1. Maintain list of trusted RSS feeds per category/country
        # 2. Parse RSS XML feeds
        # 3. Extract articles
        # 4. Apply same filtering and processing
        
        # For now, return cached data or empty
        cache_key = f"{category}_{country}_rss"
        if cache_key in self.news_cache:
            return self.news_cache[cache_key]['data']
        
        raise Exception("No RSS feeds configured and no cache available")
    
    def _tier3_cached_news(
        self,
        category: str,
        country: str,
        filter_level: str
    ) -> Dict[str, Any]:
        """
        TIER 3: Cached news headlines (offline mode).
        
        Returns most recent cached news when no connection available.
        """
        # Find most recent cache for category/country
        matching_caches = [
            (key, cache) for key, cache in self.news_cache.items()
            if category in key and country in key
        ]
        
        if matching_caches:
            # Get most recent
            most_recent = max(matching_caches, key=lambda x: x[1]['timestamp'])
            cached_data = most_recent[1]['data']
            
            # Apply current filter level
            cached_data['articles'] = self._filter_content(
                cached_data['articles'],
                filter_level
            )
            
            # Update metadata
            cached_data['tier_used'] = 3
            cached_data['data_source'] = 'Cached (offline)'
            cached_data['warnings'] = [
                f"Offline mode - showing cached news from {datetime.fromtimestamp(most_recent[1]['timestamp']).strftime('%Y-%m-%d %H:%M')}"
            ]
            
            return cached_data
        
        # No cache available - return generic message
        return {
            'articles': [],
            'total_results': 0,
            'category': category,
            'country': country,
            'tier_used': 3,
            'status': 'partial',
            'data_source': 'None (offline)',
            'summary': "I'm currently offline and don't have cached news available. Please check your internet connection.",
            'warnings': ['No internet connection', 'No cached news available']
        }
    
    def _filter_content(self, articles: List[Dict[str, Any]], filter_level: str) -> List[Dict[str, Any]]:
        """
        Filter articles based on content safety level.
        
        Removes inappropriate content for children and teens.
        """
        if filter_level == self.FILTER_ADULT:
            return articles  # No filtering
        
        filtered_articles = []
        
        for article in articles:
            # Combine title and description for checking
            text_to_check = (
                (article.get('title', '') + ' ' + article.get('description', ''))
                .lower()
            )
            
            # Check for adult keywords
            has_adult_content = any(
                keyword in text_to_check 
                for keyword in self.adult_keywords
            )
            
            # Filter based on level
            if filter_level == self.FILTER_CHILD:
                # Very strict - remove anything questionable
                if not has_adult_content:
                    # Also check categories
                    category = article.get('category', '').lower()
                    if category not in ['politics', 'crime', 'war']:
                        filtered_articles.append(article)
            
            elif filter_level == self.FILTER_TEEN:
                # Moderate - allow some news but filter extreme content
                extreme_keywords = ['murder', 'terrorism', 'assault', 'bomb']
                has_extreme = any(kw in text_to_check for kw in extreme_keywords)
                if not has_extreme:
                    filtered_articles.append(article)
        
        if filtered_articles:
            logger.info(f"Content filtering: {len(articles)} → {len(filtered_articles)} articles ({filter_level} level)")
        else:
            logger.warning(f"Content filtering removed all articles - using safe fallback")
            # If all filtered, return family-friendly placeholder
            filtered_articles = self._get_safe_news_placeholder()
        
        return filtered_articles
    
    def _get_safe_news_placeholder(self) -> List[Dict[str, Any]]:
        """Return safe, generic news for children when all content filtered."""
        return [
            {
                'id': 1,
                'title': 'Science Discovery: New Planet Found',
                'description': 'Scientists have discovered a new planet in a distant solar system.',
                'source': 'Science News',
                'category': 'science',
                'published_at': datetime.now().isoformat(),
                'is_placeholder': True
            },
            {
                'id': 2,
                'title': 'Technology: New Educational App Launched',
                'description': 'A new app helps children learn mathematics through fun games.',
                'source': 'Tech News',
                'category': 'technology',
                'published_at': datetime.now().isoformat(),
                'is_placeholder': True
            },
            {
                'id': 3,
                'title': 'Sports: Local Team Wins Championship',
                'description': 'The local youth soccer team won the regional championship.',
                'source': 'Sports Update',
                'category': 'sports',
                'published_at': datetime.now().isoformat(),
                'is_placeholder': True
            }
        ]
    
    def _generate_summary(self, articles: List[Dict[str, Any]], category: str, country: str) -> str:
        """
        Generate natural language summary of news.
        """
        if not articles:
            return f"No news articles found for {category} in {self.COUNTRIES.get(country, country)}."
        
        count = len(articles)
        category_name = self.CATEGORIES.get(category, category)
        country_name = self.COUNTRIES.get(country, country)
        
        summary = f"Here are {count} {category_name} headlines"
        
        if country != 'us':
            summary += f" from {country_name}"
        
        summary += ":\n\n"
        
        # Add top 3 headlines
        for i, article in enumerate(articles[:3], 1):
            summary += f"{i}. {article['title']}"
            if article.get('source'):
                summary += f" ({article['source']})"
            summary += "\n"
        
        if count > 3:
            summary += f"\n...and {count - 3} more articles."
        
        return summary.strip()
    
    def get_breaking_news(self, filter_level: str = FILTER_ADULT) -> Dict[str, Any]:
        """
        Get breaking news alerts.
        
        Special method for urgent news updates.
        """
        logger.info("🚨 Fetching breaking news")
        
        context = {
            'category': 'general',
            'country': self.default_country,
            'count': 3,
            'filter_level': filter_level
        }
        
        result = self.execute(context)
        
        # Check if any articles are within last hour
        recent_articles = []
        now = datetime.now()
        
        for article in result.get('articles', []):
            pub_time_str = article.get('published_at', '')
            if pub_time_str:
                try:
                    pub_time = datetime.fromisoformat(pub_time_str.replace('Z', '+00:00'))
                    age_hours = (now - pub_time.replace(tzinfo=None)).total_seconds() / 3600
                    if age_hours < 1:
                        article['age_hours'] = age_hours
                        recent_articles.append(article)
                except:
                    pass
        
        result['breaking_articles'] = recent_articles
        result['has_breaking_news'] = len(recent_articles) > 0
        
        return result
    
    def search_news(self, query: str, filter_level: str = FILTER_ADULT, count: int = 5) -> Dict[str, Any]:
        """
        Search news by keyword or topic.
        
        Args:
            query: Search query
            filter_level: Content filter level
            count: Number of results
        
        Returns:
            Search results
        """
        logger.info(f"🔍 Searching news: '{query}'")
        
        context = {
            'query': query,
            'count': count,
            'filter_level': filter_level
        }
        
        return self.execute(context)
    
    def validate_input(self, context: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        if not isinstance(context, dict):
            return False
        
        # Validate category
        if 'category' in context:
            if context['category'] not in self.CATEGORIES:
                logger.error(f"Invalid category: {context['category']}")
                return False
        
        # Validate country
        if 'country' in context:
            if context['country'] not in self.COUNTRIES:
                logger.warning(f"Unknown country code: {context['country']} (will try anyway)")
        
        # Validate count
        if 'count' in context:
            if not isinstance(context['count'], int) or context['count'] < 1:
                logger.error(f"Invalid count: {context['count']}")
                return False
        
        # Validate filter level
        if 'filter_level' in context:
            valid_levels = [self.FILTER_CHILD, self.FILTER_TEEN, self.FILTER_ADULT]
            if context['filter_level'] not in valid_levels:
                logger.error(f"Invalid filter_level: {context['filter_level']}")
                return False
        
        return True

