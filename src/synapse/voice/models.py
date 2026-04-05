from django.db import models
from django.utils import timezone

class ConversationSession(models.Model):
    """Stores conversation metadata"""
    session_id = models.CharField(max_length=255, unique=True)
    started_at = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-started_at']

class MemoryRecord(models.Model):
    """Structured memory with conflict resolution"""
    ENTITY_TYPES = (
        ('location', 'Location'),
        ('preference', 'Preference'),
        ('habit', 'Habit'),
        ('fact', 'Fact'),
        ('relationship', 'Relationship'),
    )
    
    entity = models.CharField(max_length=255)  # e.g., "key", "daughter name"
    entity_type = models.CharField(max_length=50, choices=ENTITY_TYPES)
    current_value = models.TextField()
    previous_value = models.TextField(blank=True, null=True)
    confidence = models.FloatField(default=0.8)
    
    embedding_vector = models.BinaryField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['entity']),
            models.Index(fields=['updated_at']),
        ]

class ConversationTurn(models.Model):
    """Log of each conversation turn for debugging and learning"""
    session = models.ForeignKey(ConversationSession, on_delete=models.CASCADE)
    
    user_text = models.TextField()
    qwen_intent = models.JSONField()  # Qwen's decision
    gpt_response = models.TextField(blank=True, null=True)
    spoken_response = models.TextField()
    
    memory_action = models.CharField(max_length=50, blank=True)  # store, retrieve, none
    memory_updated = models.BooleanField(default=False)
    
    latency_ms = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
