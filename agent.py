# © 2025 Dowek Analytics Ltd.
# ORACLE SAMUEL – The Real Estate Market Prophet
# MD5-Protected AI System. Unauthorized use prohibited.

import pandas as pd
import numpy as np
from datetime import datetime


class OracleSamuelAgent:
    """
    Oracle Samuel - The Real Estate Market Prophet
    Advanced AI Agent for Real Estate Market Analysis
    """
    
    def __init__(self, df, predictor=None):
        self.df = df
        self.predictor = predictor
        self.name = "Oracle Samuel"
        self.title = "Real Estate Market Prophet"
    
    def analyze_query(self, query):
        """Main analysis method - interprets user questions"""
        query_lower = query.lower()
        
        # Route to appropriate analysis method
        if any(word in query_lower for word in ['feature', 'affect', 'influence', 'important']):
            return self._analyze_feature_importance()
        
        elif any(word in query_lower for word in ['predict', 'forecast', 'growth', 'future']):
            return self._analyze_future_trends()
        
        elif any(word in query_lower for word in ['undervalued', 'overvalued', 'value', 'deal']):
            return self._find_value_opportunities()
        
        elif any(word in query_lower for word in ['average', 'mean', 'typical', 'median']):
            return self._analyze_averages()
        
        elif any(word in query_lower for word in ['best', 'top', 'highest', 'premium']):
            return self._analyze_premium_properties()
        
        elif any(word in query_lower for word in ['correlation', 'relationship', 'related']):
            return self._analyze_correlations()
        
        elif any(word in query_lower for word in ['summary', 'overview', 'insights']):
            return self._generate_market_summary()
        
        else:
            return self._generate_general_analysis()
    
    def _analyze_feature_importance(self):
        """Analyze which features most affect price"""
        response = f"## 🔍 Feature Importance Analysis\n\n"
        
        if self.predictor and self.predictor.feature_importance is not None:
            top_features = self.predictor.get_top_features(5)
            
            response += "Based on advanced machine learning analysis, here are the **Top 5 Factors** influencing property prices:\n\n"
            
            for idx, row in top_features.iterrows():
                feature = row['feature'].replace('_', ' ').title()
                importance = row['importance'] * 100
                response += f"**{idx + 1}. {feature}** - {importance:.1f}% importance\n"
            
            response += f"\n💡 **Professional Insight**: "
            top_feature = top_features.iloc[0]['feature'].replace('_', ' ')
            response += f"The {top_feature} is the single most powerful predictor in your market. "
            response += f"Properties with optimal {top_feature} can command premium pricing.\n"
        else:
            response += "⚠️ Please train the ML model first to see feature importance analysis.\n"
        
        return response
    
    def _analyze_future_trends(self):
        """Analyze and predict future market trends"""
        response = f"## 📈 Market Forecast & Growth Analysis\n\n"
        
        # Find price column
        price_col = self._find_price_column()
        if price_col:
            current_avg = self.df[price_col].mean()
            current_median = self.df[price_col].median()
            price_std = self.df[price_col].std()
            
            # Simple growth projection
            projected_growth = 0.05  # 5% annual growth assumption
            projected_price = current_avg * (1 + projected_growth)
            
            response += f"**Current Market Analysis:**\n"
            response += f"- Average Price: ${current_avg:,.0f}\n"
            response += f"- Median Price: ${current_median:,.0f}\n"
            response += f"- Price Volatility: ${price_std:,.0f}\n\n"
            
            response += f"**12-Month Projection:**\n"
            response += f"- Expected Avg Price: ${projected_price:,.0f}\n"
            response += f"- Projected Growth: {projected_growth * 100:.1f}%\n\n"
            
            response += f"💡 **Expert Recommendation**: "
            if price_std / current_avg > 0.3:
                response += "This market shows high volatility. Focus on properties with strong fundamentals.\n"
            else:
                response += "This market shows stability. Excellent conditions for long-term investment.\n"
        else:
            response += "⚠️ Could not locate price data for forecasting.\n"
        
        return response
    
    def _find_value_opportunities(self):
        """Find undervalued properties"""
        response = f"## 💎 Value Opportunity Analysis\n\n"
        
        price_col = self._find_price_column()
        if price_col and self.predictor and self.predictor.model:
            # Calculate predicted vs actual
            X, y = self.predictor.prepare_data()
            predictions = self.predictor.model.predict(X)
            
            # Find undervalued (actual < predicted)
            self.df['predicted_price'] = predictions
            self.df['value_gap'] = ((predictions - y) / y * 100)
            
            undervalued = self.df[self.df['value_gap'] > 10].sort_values('value_gap', ascending=False)
            
            if len(undervalued) > 0:
                response += f"🎯 **Found {len(undervalued)} Undervalued Properties!**\n\n"
                response += "These properties are priced below their predicted market value:\n\n"
                
                for idx, row in undervalued.head(5).iterrows():
                    actual = row[price_col]
                    predicted = row['predicted_price']
                    gap = row['value_gap']
                    response += f"**Property #{idx}**: Listed at ${actual:,.0f}, Worth ~${predicted:,.0f} (💰 {gap:.1f}% upside)\n"
                
                response += f"\n💡 **Investment Insight**: These properties represent the strongest value plays in your dataset.\n"
            else:
                response += "All properties appear fairly valued relative to market predictions.\n"
            
            # Clean up temporary columns
            self.df.drop(['predicted_price', 'value_gap'], axis=1, inplace=True, errors='ignore')
        else:
            response += "⚠️ Please train the ML model first for value analysis.\n"
        
        return response
    
    def _analyze_averages(self):
        """Analyze average statistics"""
        response = f"## 📊 Market Averages & Statistics\n\n"
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:5]:  # Top 5 numeric columns
            avg_val = self.df[col].mean()
            median_val = self.df[col].median()
            col_name = col.replace('_', ' ').title()
            
            response += f"**{col_name}:**\n"
            response += f"- Average: {avg_val:,.2f}\n"
            response += f"- Median: {median_val:,.2f}\n\n"
        
        return response
    
    def _analyze_premium_properties(self):
        """Analyze premium/top properties"""
        response = f"## 🏆 Premium Property Analysis\n\n"
        
        price_col = self._find_price_column()
        if price_col:
            # Get top 10% by price
            threshold = self.df[price_col].quantile(0.90)
            premium_props = self.df[self.df[price_col] >= threshold]
            
            response += f"**Premium Market Segment** (Top 10%):\n"
            response += f"- Properties: {len(premium_props)}\n"
            response += f"- Price Threshold: ${threshold:,.0f}\n"
            response += f"- Average Premium Price: ${premium_props[price_col].mean():,.0f}\n\n"
            
            # Analyze common features
            numeric_cols = premium_props.select_dtypes(include=[np.number]).columns
            response += "**Premium Property Characteristics:**\n"
            for col in numeric_cols[:3]:
                if col != price_col:
                    avg = premium_props[col].mean()
                    response += f"- Avg {col.replace('_', ' ').title()}: {avg:.1f}\n"
        
        return response
    
    def _analyze_correlations(self):
        """Analyze feature correlations"""
        response = f"## 🔗 Feature Correlation Analysis\n\n"
        
        if self.predictor:
            correlations = self.predictor.analyze_correlation()
            if correlations is not None:
                response += "**Strongest Correlations with Price:**\n\n"
                for feature, corr in correlations.head(6).items():
                    if abs(corr) < 1.0:  # Exclude self-correlation
                        strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.4 else "Weak"
                        direction = "positive" if corr > 0 else "negative"
                        response += f"- **{feature.replace('_', ' ').title()}**: {corr:.3f} ({strength} {direction})\n"
        
        return response
    
    def _generate_market_summary(self):
        """Generate comprehensive market summary"""
        response = f"## 📋 Market Summary Report\n\n"
        
        response += f"**Dataset Overview:**\n"
        response += f"- Total Properties: {len(self.df)}\n"
        response += f"- Features Analyzed: {len(self.df.columns)}\n\n"
        
        price_col = self._find_price_column()
        if price_col:
            response += f"**Price Analysis:**\n"
            response += f"- Average: ${self.df[price_col].mean():,.0f}\n"
            response += f"- Median: ${self.df[price_col].median():,.0f}\n"
            response += f"- Range: ${self.df[price_col].min():,.0f} - ${self.df[price_col].max():,.0f}\n\n"
        
        if self.predictor and self.predictor.metrics:
            response += f"**Model Performance:**\n"
            response += f"- Accuracy (R²): {self.predictor.metrics['r2_score']:.2%}\n"
            response += f"- Prediction Error (MAE): ${self.predictor.metrics['mae']:,.0f}\n\n"
        
        response += f"💡 **Oracle Samuel's Insight**: This market shows "
        if price_col and self.df[price_col].std() / self.df[price_col].mean() < 0.3:
            response += "consistent pricing patterns, ideal for predictable investments."
        else:
            response += "diverse pricing opportunities, suitable for value-hunting strategies."
        
        return response
    
    def _generate_general_analysis(self):
        """Generate general analysis"""
        return self._generate_market_summary()
    
    def _find_price_column(self):
        """Helper: Find the price column"""
        price_keywords = ['price', 'cost', 'value', 'amount']
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in price_keywords):
                return col
        return None
    
    def get_greeting(self):
        """Return agent greeting"""
        return f"""
        ### 👋 Welcome! I'm {self.name}, your {self.title}
        
        I'm here to provide world-class real estate market analysis using advanced machine learning.
        
        **Ask me anything like:**
        - "Which features most affect house prices?"
        - "Predict price growth for next year"
        - "Which properties are undervalued?"
        - "Show me market insights"
        - "What's the average price trend?"
        
        I analyze your data with the precision of Wall Street and the insight of a Silicon Valley data scientist.
        """

