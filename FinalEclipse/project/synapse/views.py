import logging
import os
import uuid
from datetime import timedelta

from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_http_methods, require_POST

from .models import Profile, ScanResult
from .utils import get_risk_level

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


# ----------------------------------------------------------------- pages

def landing(request):
    return render(request, 'landing.html')


@login_required
def dashboard(request):
    return render(request, 'dashboard.html')


# ------------------------------------------------------------------ auth

@require_http_methods(['GET', 'POST'])
def signup_view(request):
    if request.method != 'POST':
        return redirect('/')

    username = (request.POST.get('signup_username') or '').strip()
    password = request.POST.get('signup_password') or ''
    name = (request.POST.get('name') or '').strip()
    email = (request.POST.get('email') or '').strip()
    gender = (request.POST.get('gender') or '').strip()
    age = request.POST.get('age')

    if not username or not password:
        return render(request, 'landing.html', {'error': 'Username and password are required.'})

    if User.objects.filter(username=username).exists():
        return render(request, 'landing.html', {'error': 'That username is already taken.'})

    try:
        # create_user does not run the configured validators on its own.
        validate_password(password)
    except ValidationError as exc:
        return render(request, 'landing.html', {'error': ' '.join(exc.messages)})

    try:
        # A non-numeric age used to reach the IntegerField and raise a 500.
        age = int(age)
    except (TypeError, ValueError):
        return render(request, 'landing.html', {'error': 'Please enter your age as a number.'})
    if not 0 < age < 130:
        return render(request, 'landing.html', {'error': 'Please enter a valid age.'})

    with transaction.atomic():
        user = User.objects.create_user(username=username, password=password, email=email)
        Profile.objects.create(user=user, name=name or username, gender=gender, age=age)

    login(request, user)
    return redirect('/dashboard/')


@require_http_methods(['GET', 'POST'])
def login_view(request):
    if request.method != 'POST':
        return redirect('/')

    username = (request.POST.get('username') or '').strip()
    password = request.POST.get('password') or ''
    user = authenticate(request, username=username, password=password)

    if user is None:
        return render(request, 'landing.html', {'error': 'Incorrect username or password.'})

    login(request, user)
    return redirect('/dashboard/')


@require_POST
def logout_view(request):
    logout(request)
    # This used to fall off the end and return None on any non-POST request,
    # which Django turns into a 500.
    return redirect('/')


# ----------------------------------------------------------------- scans

def _reject(message, status=400):
    return JsonResponse({'error': message}, status=status)


def _store_upload(upload, allowed_extensions):
    """Write an upload to MEDIA_ROOT under a generated name.

    The client-supplied filename was used directly, so two people scanning
    files with the same name overwrote each other and one request deleted the
    other's file mid-inference.
    """
    extension = os.path.splitext(upload.name)[1].lower()
    if extension not in allowed_extensions:
        return None, f"Unsupported file type '{extension or upload.name}'."
    if upload.size > MAX_UPLOAD_BYTES:
        return None, 'That file is too large.'

    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    path = os.path.join(settings.MEDIA_ROOT, f"{uuid.uuid4().hex}{extension}")
    with open(path, 'wb+') as handle:
        for chunk in upload.chunks():
            handle.write(chunk)
    return path, None


def _reliability(scan_type):
    """What this model's output is actually worth, from its model card.

    Returned with every scan so the interface can state the limits rather
    than presenting a probability as a finding.
    """
    if scan_type != 'AUDIO':
        return {
            'validated': False,
            'note': 'This model has not been evaluated on a held-out set.',
        }

    try:
        from synapse.app.data.predict import get_model_card
        card = get_model_card()
    except Exception:
        card = {}

    metrics = card.get('metrics_at_threshold') or {}
    if not metrics:
        return {'validated': False, 'note': 'No evaluation recorded for this model.'}

    bands = card.get('holdout_bands') or card.get('bands') or {}
    return {
        'validated': True,
        'roc_auc': metrics.get('roc_auc'),
        'evaluation': card.get('evaluation', ''),
        # How often the model commits at all, and how often it is right when
        # it does. These are the numbers that describe a three-band output;
        # sensitivity alone describes a binary one it no longer produces.
        'answers_share': round((bands.get('share_answered') or 0) * 100, 1),
        'accuracy_when_answered': round((bands.get('accuracy_when_answered') or 0) * 100, 1),
        'base_rate': round((bands.get('base_rate') or 0) * 100, 1),
        'note': (
            'Indicative only, not a diagnosis. Measured on held-out speakers '
            'from a small corpus of recorded interviews.'
        ),
    }


def _run_scan(request, field, allowed_extensions, scan_type, predictor):
    upload = request.FILES.get(field)
    if upload is None:
        return _reject(f"No {field} file was uploaded.")

    path, error = _store_upload(upload, allowed_extensions)
    if error:
        return _reject(error)

    try:
        result, confidence = predictor(path)
    except Exception as exc:
        logger.exception(f"{scan_type} inference failed: {exc}")
        return _reject('Could not analyse that file.', status=502)
    finally:
        # Previously only removed on the happy path, so every failed scan
        # left its upload behind in MEDIA_ROOT.
        try:
            os.remove(path)
        except OSError:
            logger.warning(f"Could not remove temporary upload {path}")

    if result is None or confidence is None:
        # predict_audio returns (None, None) for unreadable audio; rounding
        # that used to raise a TypeError and return a 500.
        return _reject('Could not read that recording. Please try another file.', status=422)

    risk = get_risk_level(result, scan_type)
    if risk is None:
        logger.error(f"{scan_type} model returned an unmapped label: {result!r}")
        return _reject('Could not interpret the scan result.', status=502)

    ScanResult.objects.create(
        user=request.user,
        scan_type=scan_type,
        result=result,
        confidence=confidence,
        risk_level=risk,
    )

    return JsonResponse({
        'result': result,
        # Both models report a percentage. Audio returned 0-1 and MRI returned
        # 0-100, and the dashboard rendered them under the same label.
        'confidence': round(confidence * 100, 1),
        'risk': risk,
        'reliability': _reliability(scan_type),
    })


@login_required
@require_POST
def audio_scan(request):
    from synapse.app.data.predict import predict_audio

    return _run_scan(request, 'audio', AUDIO_EXTENSIONS, 'AUDIO', predict_audio)


@login_required
@require_POST
def mri_scan(request):
    from synapse.predict import predict_mri

    return _run_scan(request, 'mri', IMAGE_EXTENSIONS, 'MRI', predict_mri)


# ------------------------------------------------------------- dashboard

@login_required
@require_http_methods(['GET'])
def dashboard_data(request):
    scans = ScanResult.objects.filter(user=request.user).order_by('-created_at')
    total_sessions = scans.count()

    latest = scans.first()
    risk = latest.risk_level if latest else 'LOW'

    today = timezone.localdate()
    # A streak counted back from today, so someone who scanned every day for a
    # week but not yet today was shown a streak of zero. Start from the most
    # recent day that actually has a scan.
    # localtime() first: created_at is stored in UTC, and east of Greenwich its
    # .date() is the previous day. Without this, a scan at half past one in the
    # morning counted against yesterday, merging two local days into one and
    # under-reporting the streak.
    scan_days = {
        timezone.localtime(value).date()
        for value in scans.values_list('created_at', flat=True)
    }
    streak = 0
    if scan_days:
        cursor = today if today in scan_days else today - timedelta(days=1)
        while cursor in scan_days:
            streak += 1
            cursor -= timedelta(days=1)

    weekly_scores, labels = [], []
    for offset in range(6, -1, -1):
        day = today - timedelta(days=offset)
        day_scans = [
            s for s in scans if timezone.localtime(s.created_at).date() == day
        ]
        average = sum(s.confidence for s in day_scans) / len(day_scans) if day_scans else 0
        weekly_scores.append(round(average * 100, 1))
        labels.append(day.strftime('%a'))

    return JsonResponse({
        'risk': risk,
        'sessions': total_sessions,
        'streak': streak,
        'weekly_scores': weekly_scores,
        'labels': labels,
    })
