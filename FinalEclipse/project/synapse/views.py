from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from .models import Profile
from django.contrib.auth.decorators import login_required

@login_required
def dashboard(request):
    return render(request, 'dashboard.html')

def landing(request):
    return render(request, 'landing.html')



# ✅ SIGNUP VIEW (FIXED)
def signup_view(request):
    if request.method == "POST":
        username = request.POST.get('signup_username')
        password = request.POST.get('signup_password')
        name = request.POST.get('name')
        email = request.POST.get('email')
        gender = request.POST.get('gender')
        age = request.POST.get('age')

        # 🔴 Check if user already exists
        if User.objects.filter(username=username).exists():
            print("User already exists")
            return redirect('/')

        # ✅ Create User
        user = User.objects.create_user(
            username=username,
            password=password,
            email=email
        )

        # ✅ Create Profile
        Profile.objects.create(
            user=user,
            name=name,
            gender=gender,
            age=age
        )

        # ✅ Login user
        login(request, user)
        return redirect('/dashboard/')

    return redirect('/')


# ✅ LOGIN VIEW (FIXED)
def login_view(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('/dashboard/')
        else:
            print("Invalid credentials")
            return redirect('/')

    return redirect('/')



from django.contrib.auth import logout

def logout_view(request):
    if request.method == "POST":
        logout(request)
        return redirect('/')   # back to landing page
    



import os
from django.http import JsonResponse
from django.conf import settings


from .models import ScanResult
from .utils import get_risk_level

def audio_scan(request):
    if request.method == "POST" and request.FILES.get("audio"):
        audio = request.FILES["audio"]

        file_path = os.path.join(settings.MEDIA_ROOT, audio.name)

        with open(file_path, "wb+") as f:
            for chunk in audio.chunks():
                f.write(chunk)

        from synapse.app.data.predict import predict_audio
        result, confidence = predict_audio(file_path)

        os.remove(file_path)

        risk = get_risk_level(result)

        # ✅ SAVE
        ScanResult.objects.create(
            user=request.user,
            scan_type="AUDIO",
            result=result,
            confidence=confidence,
            risk_level=risk
        )

        return JsonResponse({
            "result": result,
            "confidence": round(confidence, 3),
            "risk": risk
        })

    return JsonResponse({"error": "Invalid request"}, status=400)



import os
from django.http import JsonResponse
from django.conf import settings


def mri_scan(request):
    if request.method == "POST" and request.FILES.get("mri"):
        mri_file = request.FILES["mri"]

        file_path = os.path.join(settings.MEDIA_ROOT, mri_file.name)

        with open(file_path, "wb+") as f:
            for chunk in mri_file.chunks():
                f.write(chunk)

        from synapse.predict import predict_mri
        result, confidence = predict_mri(file_path)

        os.remove(file_path)

        risk = get_risk_level(result)

        # ✅ SAVE
        ScanResult.objects.create(
            user=request.user,
            scan_type="MRI",
            result=result,
            confidence=confidence,
            risk_level=risk
        )

        return JsonResponse({
            "result": result,
            "confidence": round(confidence * 100, 2),
            "risk": risk
        })

    return JsonResponse({"error": "Invalid request"})



from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .models import ScanResult
from django.utils import timezone
from datetime import timedelta
from django.db.models import Count

@login_required
def dashboard_data(request):
    user = request.user
    scans = ScanResult.objects.filter(user=user).order_by('-created_at')
    total_sessions = scans.count()
    # ---------- RISK LEVEL ----------
    latest_scan = scans.first()
    risk = latest_scan.risk_level if latest_scan else "LOW"
    # ---------- STREAK ----------
    today = timezone.now().date()
    streak = 0
    for i in range(7):
        day = today - timedelta(days=i)
        if scans.filter(created_at__date=day).exists():
            streak += 1
        else:
            break
    # ---------- WEEKLY DATA ----------
    last_7_days = []
    labels = []
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        day_scans = scans.filter(created_at__date=day)
        if day_scans.exists():
            avg = sum([s.confidence for s in day_scans]) / day_scans.count()
        else:
            avg = 0
        last_7_days.append(round(avg, 2))
        labels.append(day.strftime('%a'))

    # ---------- RESPONSE ----------
    return JsonResponse({
        "risk": risk,
        "sessions": total_sessions,
        "streak": streak,
        "weekly_scores": last_7_days,
        "labels": labels
    })