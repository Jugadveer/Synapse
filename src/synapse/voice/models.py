from django.db import models
from django.utils import timezone


class ConversationSession(models.Model):
    """One websocket conversation."""

    session_id = models.CharField(max_length=255, unique=True)
    # Stable per-person key (the Django user id, or the anonymous session key).
    user_key = models.CharField(max_length=255, db_index=True, null=True, blank=True)
    started_at = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-started_at']

    def __str__(self):
        return f"session {self.session_id}"


class MemoryRecord(models.Model):
    """A structured thing the assistant has been asked to remember."""

    ENTITY_TYPES = (
        ('location', 'Location'),
        ('preference', 'Preference'),
        ('habit', 'Habit'),
        ('fact', 'Fact'),
        ('relationship', 'Relationship'),
    )

    # Without this every person shared one pool of memories.
    user_key = models.CharField(max_length=255, db_index=True, null=True, blank=True)
    entity = models.CharField(max_length=255)
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
            models.Index(fields=['user_key', 'entity']),
        ]

    def __str__(self):
        return f"{self.entity}: {self.current_value}"


class Reminder(models.Model):
    """A reminder the assistant promised to deliver.

    The assistant used to answer "I will remind you at 3:45" and then only
    write a sentence into memory - there was no scheduler anywhere in the
    project, so no reminder was ever delivered. Reminders are rows now, so
    they outlive the websocket that created them.
    """

    user_key = models.CharField(max_length=255, db_index=True, null=True, blank=True)
    session = models.ForeignKey(
        ConversationSession, on_delete=models.SET_NULL, null=True, blank=True
    )

    text = models.TextField(help_text="What the person asked to be reminded about.")
    spoken_time = models.CharField(
        max_length=100, blank=True, help_text="How the time was read back to them."
    )
    due_at = models.DateTimeField(db_index=True)

    delivered_at = models.DateTimeField(null=True, blank=True)
    cancelled = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['due_at']
        indexes = [
            models.Index(fields=['user_key', 'due_at']),
            models.Index(fields=['delivered_at', 'due_at']),
        ]

    def __str__(self):
        return f"{self.text} at {self.due_at:%Y-%m-%d %H:%M}"

    @property
    def is_pending(self):
        return self.delivered_at is None and not self.cancelled

    def mark_delivered(self):
        self.delivered_at = timezone.now()
        self.save(update_fields=['delivered_at'])


class ConversationTurn(models.Model):
    """One exchange, kept for debugging and for reviewing what was said."""

    session = models.ForeignKey(
        ConversationSession, on_delete=models.CASCADE, related_name='turns'
    )

    user_text = models.TextField()
    qwen_intent = models.JSONField(default=dict, blank=True)
    gpt_response = models.TextField(blank=True, null=True)
    spoken_response = models.TextField(blank=True)

    memory_action = models.CharField(max_length=50, blank=True)
    memory_updated = models.BooleanField(default=False)

    latency_ms = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']
