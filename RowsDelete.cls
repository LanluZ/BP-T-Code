VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "Sheet1"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = True
Sub Cl()
    Dim i As Long
    Dim x As Long
    i = 1
    x = 0
    While (i < 300000)
        x = 0
        If (Sheet1.Cells(i, 2) = "") Then
            Rows(i).Delete
            x = 1
        End If
        If (x = 0) Then
            i = i + 1
        End If
    Wend
End Sub

