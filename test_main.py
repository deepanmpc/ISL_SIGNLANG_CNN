#!/usr/bin/env python3
"""
Test Script for Main CNN Functionality
Tests the main functions without requiring the full dataset
"""

import torch
import torch.nn as nn
from model import SignLanguageCNN
import numpy as np

def test_model_creation():
    """Test that the model can be created successfully"""
    print("🧪 Testing model creation...")
    
    try:
        model = SignLanguageCNN(num_classes=35)
        print("✅ Model created successfully!")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 64, 64)
        output = model(dummy_input)
        
        print(f"✅ Forward pass successful!")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Expected output shape: (1, 35)")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {str(e)}")
        return False

def test_model_parameters():
    """Test model parameter counting"""
    print("\n🧪 Testing model parameters...")
    
    try:
        model = SignLanguageCNN(num_classes=35)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Parameter counting successful!")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        # Verify expected parameter count (approximately 4.3M)
        if total_params > 4_000_000 and total_params < 5_000_000:
            print(f"✅ Parameter count is in expected range (~4.3M)")
        else:
            print(f"⚠️  Parameter count ({total_params:,}) is outside expected range")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter counting failed: {str(e)}")
        return False

def test_model_layers():
    """Test that all expected layers are present"""
    print("\n🧪 Testing model layers...")
    
    try:
        model = SignLanguageCNN(num_classes=35)
        
        expected_layers = [
            'conv1', 'conv2', 'conv3', 'dropout', 'fc'
        ]
        
        model_attributes = dir(model)
        missing_layers = []
        
        for layer in expected_layers:
            if layer in model_attributes:
                print(f"✅ Layer '{layer}' found")
            else:
                print(f"❌ Layer '{layer}' missing")
                missing_layers.append(layer)
        
        if not missing_layers:
            print("✅ All expected layers present!")
            return True
        else:
            print(f"❌ Missing layers: {missing_layers}")
            return False
            
    except Exception as e:
        print(f"❌ Layer testing failed: {str(e)}")
        return False

def test_data_transforms():
    """Test data transformation functions"""
    print("\n🧪 Testing data transforms...")
    
    try:
        from torchvision import transforms
        
        # Test basic transform
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create dummy image data
        dummy_image = torch.randn(3, 100, 100)  # Simulate PIL image
        dummy_image = transforms.ToPILImage()(dummy_image)
        
        # Apply transform
        transformed = transform(dummy_image)
        
        print(f"✅ Data transforms work!")
        print(f"   Input shape: {dummy_image.size}")
        print(f"   Output shape: {transformed.shape}")
        print(f"   Expected output shape: (3, 64, 64)")
        
        return True
        
    except Exception as e:
        print(f"❌ Data transforms failed: {str(e)}")
        return False

def test_training_components():
    """Test training-related components"""
    print("\n🧪 Testing training components...")
    
    try:
        model = SignLanguageCNN(num_classes=35)
        
        # Test loss function
        criterion = nn.CrossEntropyLoss()
        dummy_output = torch.randn(2, 35)
        dummy_labels = torch.randint(0, 35, (2,))
        loss = criterion(dummy_output, dummy_labels)
        
        print(f"✅ Loss function works!")
        print(f"   Loss value: {loss.item():.4f}")
        
        # Test optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        print(f"✅ Optimizer created successfully!")
        
        # Test learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        print(f"✅ Learning rate scheduler created successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Training components failed: {str(e)}")
        return False

def test_prediction_function():
    """Test prediction functionality"""
    print("\n🧪 Testing prediction function...")
    
    try:
        model = SignLanguageCNN(num_classes=35)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 64, 64)
        
        # Make prediction
        with torch.no_grad():
            output = model(dummy_input)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        print(f"✅ Prediction function works!")
        print(f"   Predicted class: {predicted_class}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction function failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 TESTING MAIN CNN FUNCTIONALITY")
    print("=" * 60)
    
    tests = [
        test_model_creation,
        test_model_parameters,
        test_model_layers,
        test_data_transforms,
        test_training_components,
        test_prediction_function
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The main functionality is ready.")
        print("✅ You can now run the main training script.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("❌ The main functionality may not work correctly.")
    
    return passed == total

if __name__ == "__main__":
    main()
